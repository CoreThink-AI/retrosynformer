"""rs-evaluate: run retrosynthesis evaluation against all test molecules.

Accepts either an HTTP endpoint URL or a local model.pth path.

Usage::

    # Endpoint mode (API key from RETROSYNFORMER_API_KEY env var or --api-key)
    rs-evaluate --endpoint https://retrosynformer-inference-v3-knq67derjq-uc.a.run.app \\
                --study-name v3 --trial-num 000

    # Local model mode (requires matching config.yaml in the model directory)
    rs-evaluate --model results/hypertune-large-23-layer/trial_000/model.pth

Output:
    data/test_molecules_retrosynformer_{study-name}-trial{trial_num}-routes.yml
"""
import argparse
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
_PUBCHEM_PROPS = "IsomericSMILES,CanonicalSMILES,InChI,InChIKey,MolecularWeight"

# Progressive retry passes: (max_routes, max_steps, client_timeout_s, label).
# max_routes ≤ 50, max_steps ≤ 20 (endpoint schema limits).
# Client timeouts should be < Cloud Run request timeout (see rs-upload --deploy).
_RETRY_PASSES = [
    (10,  6,  240,  "pass 1"),
    (30, 10,  600,  "pass 2"),
    (50, 15, 1500,  "pass 3"),
]


# ── PubChem lookup ────────────────────────────────────────────────────────────

def _pubchem_fetch(identifier: str, id_type: str = "name") -> dict:
    """Return a dict of PubChem properties for one compound.  Empty dict on failure."""
    import requests
    encoded = urllib.parse.quote(identifier, safe="")
    url = f"{_PUBCHEM_BASE}/{id_type}/{encoded}/property/{_PUBCHEM_PROPS}/JSON"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return {}
        rows = r.json().get("PropertyTable", {}).get("Properties", [])
        if not rows:
            return {}
        p = rows[0]
        return {
            "pubchem_cid": p.get("CID"),
            "smiles": p.get("IsomericSMILES"),
            "canonical_smiles": p.get("CanonicalSMILES"),
            "inchi": p.get("InChI"),
            "inchikey": p.get("InChIKey"),
            "mol_weight": p.get("MolecularWeight"),
        }
    except Exception:
        return {}


def pubchem_lookup(mol: dict) -> dict:
    """Fill in null fields in a molecule dict from PubChem. Returns updated copy."""
    mol = dict(mol)
    # Skip lookup if we already have everything we need
    if mol.get("smiles") and mol.get("inchikey"):
        return mol

    result: dict = {}
    if mol.get("pubchem_cid"):
        result = _pubchem_fetch(str(mol["pubchem_cid"]), "cid")
    if not result and mol.get("smiles"):
        result = _pubchem_fetch(mol["smiles"], "smiles")
    if not result and mol.get("pubchem_name"):
        result = _pubchem_fetch(mol["pubchem_name"], "name")
    if not result and mol.get("query_name"):
        result = _pubchem_fetch(mol["query_name"], "name")

    for key, val in result.items():
        if val is not None and mol.get(key) is None:
            mol[key] = val
    return mol


# ── SMILES validation ─────────────────────────────────────────────────────────

def smiles_valid(smiles: str | None) -> bool:
    if not smiles:
        return False
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


# ── Route-fetching: endpoint mode ─────────────────────────────────────────────

def fetch_routes_endpoint(
    smiles: str,
    endpoint: str,
    api_key: str | None,
    max_routes: int,
    top_routes: int,
    max_steps: int = 6,
    timeout: int = 240,
) -> list[dict]:
    import requests
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = requests.post(
        f"{endpoint.rstrip('/')}/retrosynthesis",
        json={"smiles": smiles, "max_routes": max_routes, "max_steps": max_steps},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    routes = data.get("ai_routes", [])
    return [_normalise_route(r) for r in routes[:top_routes]]


def _normalise_route(r: dict) -> dict:
    """Keep only the fields we want in the YAML."""
    steps = []
    for s in r.get("steps", []):
        steps.append({
            "step": s["step"],
            "target": s["target"],
            "reaction_id": s.get("reaction_id"),
            "reactants": s.get("reactants", []),
        })
    return {
        "score": r.get("score", 0.0),
        "depth": r.get("depth", len(steps)),
        "all_leaves_purchasable": r.get("all_leaves_purchasable", False),
        "steps": steps,
        "leaf_molecules": r.get("leaf_molecules", []),
    }


# ── Route-fetching: local model mode ─────────────────────────────────────────

_local_predictor = None


def _load_local_predictor(model_path: Path, config_path: Path):
    global _local_predictor
    if _local_predictor is not None:
        return _local_predictor
    from retrosynformer.serve.predictor import ModelPredictor
    print(f"Loading model from {model_path} with config {config_path} …", flush=True)
    _local_predictor = ModelPredictor(
        config_path=str(config_path),
        model_path=str(model_path),
    )
    return _local_predictor


def fetch_routes_local(
    smiles: str, model_path: Path, config_path: Path, beam_width: int, top_routes: int
) -> list[dict]:
    predictor = _load_local_predictor(model_path, config_path)
    # predict_retrosynthesis_sync returns list[dict] in ai_routes format
    try:
        routes = predictor.predict_retrosynthesis_sync(
            smiles=smiles, max_routes=top_routes, max_steps=6
        )
    except TypeError:
        # Fallback: older API without max_routes
        routes = predictor.predict_retrosynthesis_sync(smiles)
    return [_normalise_route(r) for r in (routes or [])[:top_routes]]


# ── Study name / trial auto-detection ────────────────────────────────────────

def detect_study_trial(model_path: Path | None, endpoint: str | None) -> tuple[str, str]:
    """Return (study_name, trial_num) inferred from model path or endpoint URL."""
    if model_path:
        parts = model_path.parts
        # Look for trial_NNN directory
        trial_num = "000"
        study_name = "model"
        for i, p in enumerate(parts):
            m = re.match(r"trial[_-]?(\d+)", p, re.IGNORECASE)
            if m:
                trial_num = m.group(1).zfill(3)
                # Study name is the parent directory of the trial dir
                if i > 0:
                    study_name = parts[i - 1]
                break
        # Fallback: use parent directory name
        if study_name == "model":
            study_name = model_path.parent.name or model_path.stem
        return study_name, trial_num

    if endpoint:
        # e.g. retrosynformer-inference-v3-knq67derjq-uc.a.run.app → v3
        host = urllib.parse.urlparse(endpoint).netloc or endpoint
        m = re.search(r"retrosynformer-inference-(\w+)", host)
        study_name = m.group(1) if m else "endpoint"
        return study_name, "000"

    return "unknown", "000"


# ── YAML writer (no PyYAML dependency required — writes clean YAML manually) ──

def _yaml_str(val: Any, indent: int = 0) -> str:
    """Minimal YAML renderer for dicts/lists/scalars."""
    pad = " " * indent
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "null"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        # Quote strings containing special YAML chars
        if any(c in val for c in ":#{}[]|>&*!,'\"@`") or val.startswith(" ") or val == "":
            escaped = val.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return val
    if isinstance(val, list):
        if not val:
            return "[]"
        lines = []
        for item in val:
            if isinstance(item, dict):
                sub = _yaml_dict_block(item, indent + 2)
                lines.append(f"{pad}  -" + sub)
            else:
                lines.append(f"{pad}  - {_yaml_str(item)}")
        return "\n" + "\n".join(lines)
    if isinstance(val, dict):
        return "\n" + _yaml_dict_block(val, indent + 2)
    return str(val)


def _yaml_dict_block(d: dict, indent: int) -> str:
    pad = " " * indent
    lines = []
    for k, v in d.items():
        rendered = _yaml_str(v, indent)
        if isinstance(v, (dict, list)) and v:
            lines.append(f"{pad}{k}:{rendered}")
        else:
            lines.append(f"{pad}{k}: {rendered}")
    return "\n".join(lines)


def write_yaml(molecules: list[dict], path: Path, header_lines: list[str]) -> None:
    lines = [f"# {l}" for l in header_lines] + ["", "molecules:"]
    for mol in molecules:
        lines.append(f"  - {_yaml_dict_block(mol, 4).lstrip()}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


# ── Markdown report writer ────────────────────────────────────────────────────

def write_report(molecules: list[dict], path: Path, meta: dict) -> None:
    solved = sum(
        1 for m in molecules
        if any(r.get("all_leaves_purchasable") for r in (m.get("retrosynformer_routes") or []))
    )
    trivial = sum(
        1 for m in molecules
        if any(r.get("depth", 99) == 0 for r in (m.get("retrosynformer_routes") or []))
    )
    errors = sum(1 for m in molecules if m.get("retrosynformer_error"))
    valid = len(molecules) - errors

    def best_route(mol):
        routes = mol.get("retrosynformer_routes") or []
        return routes[0] if routes else None

    cyclic_count = 0
    for mol in molecules:
        br = best_route(mol)
        if br and _is_cyclic(br):
            cyclic_count += 1

    depths = [br["depth"] for m in molecules if (br := best_route(m)) and br and br["depth"] > 0]
    avg_depth = sum(depths) / len(depths) if depths else 0

    pass_counts = [0, 0, 0]
    for m in molecules:
        p = m.get("retrosynformer_solved_on_pass")
        if p and 1 <= p <= 3:
            pass_counts[p - 1] += 1

    lines = [
        f"# RetroSynFormer Evaluation Report",
        f"",
        f"**Date:** {meta['date']}  ",
        f"**Study:** {meta['study_name']}  Trial: {meta['trial_num']}  ",
        f"**Mode:** {meta['mode']}  ",
        f"**Initial beam width (max_routes):** {meta['beam_width']}  ",
        f"**Top routes saved:** {meta['top_routes']}  ",
        f"**Progressive retry:** pass 1 ({meta['beam_width']}/6), pass 2 (30/10), pass 3 (50/15)",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Molecules tested | {valid} |",
        f"| Skipped (no SMILES / error) | {errors} |",
        f"| **Solved** (all_leaves_purchasable) | **{solved}/{valid}** |",
        f"| Solved on pass 1 (max_routes={meta['beam_width']}, max_steps=6) | {pass_counts[0]} |",
        f"| Solved on pass 2 (max_routes=30, max_steps=10) | {pass_counts[1]} |",
        f"| Solved on pass 3 (max_routes=50, max_steps=15) | {pass_counts[2]} |",
        f"| Trivially solved (depth=0, is a building block) | {trivial} |",
        f"| Cyclic best route | {cyclic_count} |",
        f"| Avg depth of non-trivial best route | {avg_depth:.1f} |",
        f"",
        f"---",
        f"",
        f"## Per-Molecule Results",
        f"",
        f"| Molecule | Complexity | Routes | Best depth | Solved | Pass | Cyclic | Leaves (purch/total) | Score |",
        f"|----------|-----------|--------|-----------|--------|------|--------|----------------------|-------|",
    ]

    for mol in molecules:
        name = mol.get("query_name", "?")
        cplx = mol.get("complexity", "—")
        err = mol.get("retrosynformer_error")
        if err:
            lines.append(f"| {name} | {cplx} | — | — | — | — | — | — | *{err}* |")
            continue
        routes = mol.get("retrosynformer_routes") or []
        br = routes[0] if routes else None
        n_routes = len(routes)
        depth = br["depth"] if br else "—"
        solved_icon = "✓" if br and br.get("all_leaves_purchasable") else "✗"
        pass_icon = str(mol.get("retrosynformer_solved_on_pass") or "—")
        cyclic_icon = "⚠" if br and _is_cyclic(br) else ""
        leaves = br.get("leaf_molecules", []) if br else []
        purch = sum(1 for l in leaves if l.get("purchasable"))
        leaf_str = f"{purch}/{len(leaves)}"
        score = f"{br['score']:.4f}" if br else "—"
        lines.append(
            f"| {name} | {cplx} | {n_routes} | {depth} | {solved_icon} | {pass_icon} | "
            f"{cyclic_icon} | {leaf_str} | {score} |"
        )

    lines += ["", "---", "", "## Per-Molecule Route Details", ""]

    for mol in molecules:
        name = mol.get("query_name", "?")
        smiles = mol.get("smiles", "—")
        cid = mol.get("pubchem_cid", "—")
        lines.append(f"### {name}  (PubChem CID: {cid})")
        lines.append(f"**SMILES:** `{smiles}`  ")
        err = mol.get("retrosynformer_error")
        if err:
            lines.append(f"*Skipped: {err}*")
            lines.append("")
            continue

        routes = mol.get("retrosynformer_routes") or []
        if not routes:
            lines.append("*No routes returned.*")
            lines.append("")
            continue

        br = routes[0]
        depth = br["depth"]
        solved = br.get("all_leaves_purchasable", False)
        score = br.get("score", 0.0)
        cyclic = _is_cyclic(br)

        lines.append(f"**Best route:** depth={depth}  solved={solved}  score={score:.4g}  "
                     f"{'⚠ cyclic' if cyclic else ''}")
        lines.append("")

        steps = br.get("steps", [])
        if not steps:
            lines.append("- *Target molecule is itself a building block (depth=0)*")
        else:
            lines.append("**Reactions (retrosynthetic direction, target → reactants):**")
            for s in steps:
                tgt = s.get("target", "?")
                rxts = s.get("reactants", [])
                lines.append(f"{s['step']}. `{tgt}` → {' + '.join(f'`{r}`' for r in rxts)}")

        leaves = br.get("leaf_molecules", [])
        if leaves:
            lines.append("")
            lines.append("**Building blocks proposed:**")
            for lf in leaves:
                mark = "✓" if lf.get("purchasable") else "✗"
                lines.append(f"- {mark} `{lf['smiles']}`")

        if cyclic:
            lines.append("")
            lines.append("> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — "
                         "beam search looped without finding purchasable leaves.")
        lines.append("")

    lines += [
        "---",
        "",
        "## Notes on Model Behavior",
        "",
        "- **score=0.0** reflects floating-point underflow of `trajectory_prob` "
          "(product of per-step probabilities across 6 steps), not zero probability. "
          "These routes are still chemically valid proposals.",
        "- **Cyclic routes** occur when the model repeatedly applies ester "
          "hydrolysis ↔ esterification or similar reversible transforms, "
          "indicating the beam search depth limit (6) was reached without finding "
          "a purchasable route.",
        "- **depth=0** means the target SMILES itself matches a known building block "
          "in the PaRoutes training set.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n")


def _is_cyclic(route: dict) -> bool:
    """True if any step target also appears as a reactant in the same route."""
    targets = {s["target"] for s in route.get("steps", [])}
    for s in route.get("steps", []):
        if any(r in targets for r in s.get("reactants", [])):
            return True
    return False


# ── .env loader ──────────────────────────────────────────────────────────────

def _load_dotenv_defaults() -> dict[str, str]:
    """Read RETROSYNFORMER_URL and RETROSYNFORMER_API_KEY from .env files.

    Searches in order: cwd, repo root (parents[3] relative to this file),
    then the adjacent synthesis-routes-generator directory.  First value wins.
    """
    want = {"RETROSYNFORMER_URL", "RETROSYNFORMER_API_KEY"}
    found: dict[str, str] = {}
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).parents[3] / ".env",  # repo root
        Path(__file__).parents[4] / "synthesis-routes-generator" / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    if key not in want:
                        continue
                    val = val.strip().strip('"').strip("'")
                    found.setdefault(key, val)
        except OSError:
            pass
        if found.keys() >= want:
            break
    return found


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import datetime

    dotenv = _load_dotenv_defaults()
    default_endpoint = os.environ.get("RETROSYNFORMER_URL") or dotenv.get("RETROSYNFORMER_URL")
    default_api_key = os.environ.get("RETROSYNFORMER_API_KEY") or dotenv.get("RETROSYNFORMER_API_KEY")

    parser = argparse.ArgumentParser(
        description="Evaluate RetroSynFormer routes against test molecules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Endpoint mode (URL defaults to RETROSYNFORMER_URL in .env)\n"
            "  rs-evaluate --study-name v3\n\n"
            "  # Override endpoint explicitly\n"
            "  rs-evaluate --endpoint https://retrosynformer-inference-v3-knq67derjq-uc.a.run.app"
            " --study-name v3\n\n"
            "  # Local model mode\n"
            "  rs-evaluate --model results/hypertune-large-23-layer/trial_000/model.pth\n\n"
            "  Progressive retry (endpoint mode only):\n"
            "    Pass 1: max_routes=10, max_steps=6\n"
            "    Pass 2: max_routes=30, max_steps=10  (if not solved)\n"
            "    Pass 3: max_routes=50, max_steps=15  (if still not solved)\n"
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=default_endpoint,
        help=(
            "HTTP endpoint base URL (default: RETROSYNFORMER_URL from .env or env var; "
            f"currently: {default_endpoint or 'not set'})"
        ),
    )
    parser.add_argument("--model", type=Path, help="Path to local model.pth")
    parser.add_argument("--config", type=Path, help="Config YAML for local model (auto-detected if omitted)")
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        help="API key for endpoint (default: RETROSYNFORMER_API_KEY from .env or env var)",
    )
    parser.add_argument(
        "--test-molecules", type=Path,
        help="Path to test_molecules.yml (default: searches standard locations)",
    )
    parser.add_argument("--beam-width", type=int, default=10,
                        help="Initial max_routes for pass 1 (default: 10); passes 2/3 use 30 and 50")
    parser.add_argument("--top-routes", type=int, default=3, help="Routes per molecule to save (default: 3)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output directory (default: data/)")
    parser.add_argument("--study-name", help="Override auto-detected study name")
    parser.add_argument("--trial-num", help="Override auto-detected trial number (e.g. 000)")
    parser.add_argument("--pubchem", action="store_true", default=True,
                        help="Fill null fields from PubChem API (default: on)")
    parser.add_argument("--no-pubchem", dest="pubchem", action="store_false",
                        help="Skip PubChem lookups")
    parser.add_argument("--report", action="store_true", default=True, help="Write markdown report (default: on)")
    parser.add_argument("--no-report", dest="report", action="store_false")
    args = parser.parse_args()

    if args.model:
        args.endpoint = None  # --model takes precedence
    if not args.endpoint and not args.model:
        parser.error(
            "Provide either --endpoint URL or --model path/to/model.pth, "
            "or set RETROSYNFORMER_URL in .env"
        )

    # Auto-detect study name and trial number
    model_path = args.model
    if model_path:
        model_path = model_path.resolve()

    study_name, trial_num = detect_study_trial(model_path, args.endpoint)
    if args.study_name:
        study_name = args.study_name
    if args.trial_num:
        trial_num = args.trial_num.zfill(3)

    # Find test molecules YAML
    test_mol_path = args.test_molecules
    if test_mol_path is None:
        candidates = [
            Path(__file__).parents[4] / "synthesis-routes-generator" / "eval" / "test_molecules.yml",
            Path("test_molecules.yml"),
            Path("data/test_molecules.yml"),
            Path("eval/test_molecules.yml"),
        ]
        for c in candidates:
            if c.exists():
                test_mol_path = c
                break
        if test_mol_path is None:
            sys.exit(
                "Could not find test_molecules.yml. "
                "Provide it with --test-molecules <path>."
            )
    print(f"Test molecules: {test_mol_path}", flush=True)

    # Load YAML (avoid PyYAML dependency — parse manually if needed, else import)
    try:
        import yaml
        with open(test_mol_path) as f:
            raw = yaml.safe_load(f)
        molecules: list[dict] = raw.get("molecules", [])
        yaml_header_comment = ""
    except ImportError:
        sys.exit("PyYAML is required: pip install pyyaml")

    print(f"Loaded {len(molecules)} molecules.", flush=True)

    # Resolve config for local model
    config_path = args.config
    if model_path and config_path is None:
        # Look for config.yaml next to or above the model
        for p in [model_path.parent / "config.yaml", model_path.parent.parent / "config.yaml",
                  Path("results/config.yaml")]:
            if p.exists():
                config_path = p
                break
        if config_path is None:
            sys.exit("Could not find config.yaml. Provide it with --config <path>.")

    # API key
    api_key = args.api_key or os.environ.get("RETROSYNFORMER_API_KEY")

    # Output paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"test_molecules_retrosynformer_{study_name}-trial{trial_num}-routes"
    out_yaml = args.output_dir / f"{out_stem}.yml"
    out_md   = args.output_dir / f"{out_stem}_report.md"

    mode = f"endpoint: {args.endpoint}" if args.endpoint else f"local model: {model_path}"
    print(f"Mode: {mode}", flush=True)
    print(f"Output: {out_yaml}", flush=True)

    # Build pass config: honour --beam-width for pass 1, then fixed escalation.
    # Each entry: (max_routes, max_steps, client_timeout_s, label)
    pass_config = [
        (args.beam_width, _RETRY_PASSES[0][1], _RETRY_PASSES[0][2], _RETRY_PASSES[0][3]),
        *_RETRY_PASSES[1:],
    ]

    # Evaluate each molecule (progressive retry in endpoint mode)
    results: list[dict] = []
    n = len(molecules)
    for i, mol in enumerate(molecules):
        name = mol.get("query_name", f"mol_{i}")
        smiles = mol.get("smiles")

        print(f"\n[{i+1}/{n}] {name}", flush=True)

        # PubChem lookup for missing fields
        if args.pubchem and (not smiles or not mol.get("inchikey")):
            print("  Looking up PubChem …", flush=True)
            mol = pubchem_lookup(mol)
            smiles = mol.get("smiles")

        if not smiles_valid(smiles):
            reason = "no SMILES" if not smiles else "invalid SMILES"
            print(f"  Skipped: {reason}", flush=True)
            mol = dict(mol)
            mol["retrosynformer_routes"] = None
            mol["retrosynformer_error"] = reason
            results.append(mol)
            continue

        routes: list[dict] = []
        solved_on_pass: int | None = None
        pass_error: str | None = None

        passes = pass_config if args.endpoint else [pass_config[0]]
        for pass_num, (max_routes, max_steps, req_timeout, pass_label) in enumerate(passes, 1):
            if pass_num > 1:
                print(
                    f"  → {pass_label}: max_routes={max_routes} max_steps={max_steps} "
                    f"timeout={req_timeout}s",
                    flush=True,
                )
            t0 = time.perf_counter()
            try:
                if args.endpoint:
                    routes = fetch_routes_endpoint(
                        smiles, args.endpoint, api_key, max_routes, args.top_routes,
                        max_steps, req_timeout,
                    )
                else:
                    routes = fetch_routes_local(
                        smiles, model_path, config_path, max_routes, args.top_routes
                    )
                elapsed = time.perf_counter() - t0
                br = routes[0] if routes else None
                solved = bool(br and br.get("all_leaves_purchasable"))
                depth = br["depth"] if br else "—"
                cyclic = _is_cyclic(br) if br else False
                print(
                    f"  {pass_label}: {len(routes)} route(s)  depth={depth}  solved={solved}  "
                    f"{'⚠cyclic ' if cyclic else ''}{elapsed:.1f}s",
                    flush=True,
                )
                if solved:
                    solved_on_pass = pass_num
                    break
                pass_error = None  # reset any earlier error
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"  ERROR ({elapsed:.1f}s): {exc}", flush=True)
                pass_error = f"API error: {exc}"
                break  # don't retry on hard errors

        mol = dict(mol)
        if pass_error and not routes:
            mol["retrosynformer_routes"] = None
            mol["retrosynformer_error"] = pass_error
        else:
            mol["retrosynformer_routes"] = routes
            mol["retrosynformer_solved_on_pass"] = solved_on_pass
        results.append(mol)

    # Write YAML
    import datetime as dt
    today = dt.date.today().isoformat()
    header = [
        f"RetroSynFormer evaluation routes",
        f"Generated: {today}",
        f"Study: {study_name}  Trial: {trial_num}",
        f"Mode: {mode}",
        f"pass1: max_routes={args.beam_width} max_steps=6  "
        f"pass2: max_routes=30 max_steps=10  "
        f"pass3: max_routes=50 max_steps=15  top_routes: {args.top_routes}",
    ]
    print(f"\nWriting {out_yaml} …", flush=True)
    try:
        import yaml

        # Merge results back into YAML preserving original field order
        out_mols = []
        _route_keys = {"retrosynformer_routes", "retrosynformer_error", "retrosynformer_solved_on_pass"}
        for mol in results:
            d = {k: v for k, v in mol.items() if k not in _route_keys}
            if mol.get("retrosynformer_error"):
                d["retrosynformer_routes"] = None
                d["retrosynformer_error"] = mol["retrosynformer_error"]
            else:
                d["retrosynformer_routes"] = mol.get("retrosynformer_routes")
                if mol.get("retrosynformer_solved_on_pass") is not None:
                    d["retrosynformer_solved_on_pass"] = mol["retrosynformer_solved_on_pass"]
            out_mols.append(d)

        header_str = "\n".join(f"# {l}" for l in header)
        body = yaml.dump(
            {"molecules": out_mols},
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )
        out_yaml.write_text(header_str + "\n\n" + body)
    except Exception as exc:
        sys.exit(f"Failed to write YAML: {exc}")

    # Write markdown report
    if args.report:
        print(f"Writing {out_md} …", flush=True)
        write_report(results, out_md, {
            "date": today,
            "study_name": study_name,
            "trial_num": trial_num,
            "mode": mode,
            "beam_width": args.beam_width,
            "top_routes": args.top_routes,
        })

    # Summary
    solved_count = sum(
        1 for m in results
        if any(r.get("all_leaves_purchasable") for r in (m.get("retrosynformer_routes") or []))
    )
    valid_count = sum(1 for m in results if m.get("retrosynformer_routes") is not None or
                      m.get("retrosynformer_error") is None)
    skipped = sum(1 for m in results if m.get("retrosynformer_error"))
    tested = len(results) - skipped
    print(f"\nDone. {tested} tested, {solved_count}/{tested} solved, {skipped} skipped.")
    print(f"YAML:   {out_yaml}")
    if args.report:
        print(f"Report: {out_md}")


if __name__ == "__main__":
    main()
