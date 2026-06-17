#!/usr/bin/env python
"""Automated patch-version bump for RetroSynFormer.

Collects git commits since the last version tag, calls the Anthropic API to
draft a changelog entry, full release notes, and an annotated tag message,
then writes the files, commits, and tags — all in one command.

Usage:
    python scripts/bump.py              # bump patch, write notes, commit, tag
    python scripts/bump.py --dry-run    # preview without writing anything
    python scripts/bump.py --push       # also push branch + tag after committing
    rs-bump / bump                      # same via installed CLI entry point

Requires ANTHROPIC_API_KEY in the environment or in .env.
Install the SDK:  uv sync --extra dev   (adds anthropic>=0.25)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # …/retrosynformer/


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], **kwargs) -> str:
    return subprocess.check_output(cmd, text=True, cwd=ROOT, **kwargs).strip()


def _git_log_since(tag: str) -> str:
    """Commit subject + stat lines since tag; falls back to last 30 commits."""
    try:
        _run(["git", "rev-parse", "--verify", tag])  # check tag exists
        return _run(["git", "log", f"{tag}..HEAD", "--oneline", "--stat"])
    except subprocess.CalledProcessError:
        return _run(["git", "log", "--oneline", "--stat", "-30"])


def _git_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _current_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError("version not found in pyproject.toml")
    return m.group(1)


def _bump_patch(version: str) -> str:
    parts = version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def _call_api(commits: str, old_version: str, new_version: str,
              branch: str, today: str, model: str) -> dict:
    try:
        import anthropic
    except ImportError:
        print(
            "ERROR: 'anthropic' package not installed.\n"
            "Run:  uv sync --extra dev",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = _load_api_key()
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not set in environment or .env",
            file=sys.stderr,
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Pull the two most recent changelog entries as style examples.
    changelog_text = (ROOT / "CHANGELOG.md").read_text()
    examples = "\n---\n".join(
        re.findall(r"## \[[\s\S]*?(?=\n---)", changelog_text)[:2]
    )

    system = (
        "You are a technical writer producing release documentation for "
        "RetroSynFormer, a retrosynthesis route planner built on a Decision "
        "Transformer (GPT-2 backbone) that iteratively selects reaction templates "
        "to decompose a target molecule into purchasable building blocks. "
        "Write concisely, technically, and in the third person. "
        "Respond with valid JSON only — no markdown fences, no extra text."
    )

    prompt = f"""Generate release documentation for RetroSynFormer {new_version}.

Today: {today}
Branch: {branch}
Commits since v{old_version}:
{commits}

Match this exact CHANGELOG style (two most recent entries):
{examples}

Return a single JSON object with these exact keys:
{{
  "changelog_headline": "One sentence bold summary (no markdown bold markers, no trailing period)",
  "changelog_bullets": ["item 1", "item 2"],
  "release_notes_summary": "2-3 sentence paragraph for the ## Summary section",
  "release_notes_sections": [
    {{
      "heading": "1. Feature Name (`changed_file.py`)",
      "body": "Markdown prose and/or lists describing this change in detail"
    }}
  ],
  "tag_line1": "One-line headline for the annotated git tag (≤72 chars)",
  "tag_body": "2-3 sentences expanding on the headline for the tag body"
}}

Rules:
- changelog_bullets: 3-6 items; backtick-quote file/function/class names
- release_notes_sections: one section per logical change grouping
- tag_line1: terse imperative summary, no trailing period
"""

    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]+\}", raw)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Could not parse JSON from API response:\n{raw}")


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def _build_release_notes(data: dict, new_version: str, branch: str,
                          month_year: str) -> str:
    lines = [
        f"# RetroSynFormer {new_version} Release Notes",
        "",
        f"*Branch: `{branch}` — {month_year}*",
        "",
        "---",
        "",
        "## Summary",
        "",
        data["release_notes_summary"],
        "",
        "---",
        "",
        "## Changes",
        "",
    ]
    for section in data["release_notes_sections"]:
        lines += [f"### {section['heading']}", "", section["body"], ""]
    return "\n".join(lines)


def _build_changelog_entry(data: dict, new_version: str, today: str) -> str:
    bullets = "\n".join(f"- {b}" for b in data["changelog_bullets"])
    return (
        f"## [{new_version}] — {today}\n\n"
        f"**{data['changelog_headline']}**\n\n"
        f"{bullets}\n\n"
        f"[Full notes](docs/release-notes-{new_version}.md)\n"
    )


def _build_tag_message(data: dict) -> str:
    return f"{data['tag_line1']}\n\n{data['tag_body']}"


# ---------------------------------------------------------------------------
# File / git mutations
# ---------------------------------------------------------------------------

def _write_version(new_version: str) -> None:
    path = ROOT / "pyproject.toml"
    text = path.read_text()
    text = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        f'\\1"{new_version}"',
        text,
        flags=re.MULTILINE,
    )
    path.write_text(text)


def _prepend_changelog(entry: str) -> None:
    path = ROOT / "CHANGELOG.md"
    text = path.read_text()
    # Insert after the header separator (first "---\n\n")
    marker = "---\n\n"
    idx = text.index(marker) + len(marker)
    path.write_text(text[:idx] + entry + "\n---\n\n" + text[idx:])


def _git_commit_and_tag(new_version: str, tag_message: str) -> None:
    files = [
        "pyproject.toml",
        f"docs/release-notes-{new_version}.md",
        "CHANGELOG.md",
    ]
    subprocess.run(["git", "add"] + files, cwd=ROOT, check=True)
    commit_msg = (
        f"chore: bump to {new_version}; release notes and changelog\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "tag", "-a", f"v{new_version}", "-m", tag_message],
        cwd=ROOT, check=True,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generated content without writing files or committing",
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Run 'git push && git push origin vX.Y.Z' after committing",
    )
    parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        metavar="MODEL",
        help="Anthropic model for drafting (default: claude-haiku-4-5-20251001)",
    )
    args = parser.parse_args()

    today = date.today().isoformat()
    month_year = date.today().strftime("%B %Y")

    old_version = _current_version()
    new_version = _bump_patch(old_version)
    branch = _git_branch()
    commits = _git_log_since(f"v{old_version}")

    if not commits.strip():
        print("No commits since last tag — nothing to bump.")
        sys.exit(0)

    print(f"Bumping {old_version} → {new_version}  [{branch}]")
    print(f"Commits since v{old_version}:\n{commits}\n")
    print(f"Calling Anthropic API ({args.model}) …")

    data = _call_api(commits, old_version, new_version, branch, today, args.model)

    release_notes = _build_release_notes(data, new_version, branch, month_year)
    changelog_entry = _build_changelog_entry(data, new_version, today)
    tag_message = _build_tag_message(data)

    if args.dry_run:
        sep = "\n" + "─" * 60 + "\n"
        print(sep + "RELEASE NOTES" + sep + release_notes)
        print(sep + "CHANGELOG ENTRY" + sep + changelog_entry)
        print(sep + "TAG MESSAGE" + sep + tag_message + "\n")
        return

    notes_path = ROOT / f"docs/release-notes-{new_version}.md"
    notes_path.write_text(release_notes)
    print(f"  wrote  {notes_path.relative_to(ROOT)}")

    _write_version(new_version)
    print(f"  bumped pyproject.toml → {new_version}")

    _prepend_changelog(changelog_entry)
    print(f"  updated CHANGELOG.md")

    _git_commit_and_tag(new_version, tag_message)
    print(f"  committed + tagged v{new_version}")

    if args.push:
        subprocess.run(["git", "push"], cwd=ROOT, check=True)
        subprocess.run(
            ["git", "push", "origin", f"v{new_version}"],
            cwd=ROOT, check=True,
        )
        print(f"  pushed branch and tag v{new_version}")
    else:
        print(f"\nNext: git push && git push origin v{new_version}")


if __name__ == "__main__":
    main()
