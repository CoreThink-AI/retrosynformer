#!/usr/bin/env python3
"""Benchmark RetroSynFormer on test_molecules.yml and generate yml/md reports.

Usage (from repo root with venv activated):
    python scripts/benchmark_test_molecules.py \\
        --model-dir results/hypertune-large-emma-24-26_layer/trial_000 \\
        --beam-width 10 \\
        --out-name large_emma_24layers_trial000
"""
import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

from retrosynformer.runner import init_model, read_config
from retrosynformer.inference import RoutePredictor


def _beam_to_route_dict(beam) -> dict:
    actions = [int(a.item()) for a in beam.predicted_actions]
    rewards = list(beam.env.rewards) if hasattr(beam.env, "rewards") else []
    leafs = list(beam.env.leafs) if hasattr(beam.env, "leafs") else []
    dead_ends = list(beam.env.dead_ends) if hasattr(beam.env, "dead_ends") else []

    reactions = []
    for i, rxn_smarts in enumerate(beam.reaction_list):
        entry = {"reaction_smarts": rxn_smarts}
        if i < len(actions):
            entry["template_index"] = actions[i]
        if i < len(rewards):
            entry["reward"] = float(rewards[i])
        reactions.append(entry)

    return {
        "route_solved": bool(beam.route_solved),
        "trajectory_prob": float(beam.trajectory_prob),
        "n_steps": len(beam.reaction_list),
        "reactions": reactions,
        "leaf_smiles": leafs,
        "dead_ends": dead_ends,
    }


def _best_epoch_metrics(model_dir: str) -> tuple[int | None, dict]:
    path = Path(model_dir) / "train_progress.jsonl"
    best_epoch, best_metrics = None, {}
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line.strip())
                if r.get("is_best"):
                    best_epoch = r["epoch"]
                    best_metrics = r
    return best_epoch, best_metrics


def _generate_md(yml_data: dict, args, out_yml: str,
                 best_epoch, best_metrics: dict) -> str:
    today = str(date.today())
    out_md = f"docs/test_molecules_routes_report_{args.out_name}.md"

    v_acc = best_metrics.get("valid_action_accuracy", 0)
    v_racc = best_metrics.get("valid_route_accuracy", 0)
    v_loss = best_metrics.get("valid_loss", 0)
    n_layers = yml_data["n_layers"]
    n_solved = yml_data["n_solved"]
    n_skipped = yml_data["n_skipped"]
    n_molecules = yml_data["n_molecules"]
    n_valid = n_molecules - n_skipped

    header = (
        f"# RetroSynFormer — {args.study} / {args.trial}\n\n"
        f"**Date:** {today}  \n"
        f"**Study:** `{args.study}`  \n"
        f"**Trial:** `{args.trial}` — best checkpoint epoch {best_epoch}"
        f" (min valid_loss={v_loss:.5f})  \n"
        f"**Architecture:** {n_layers} layers, hidden_size={yml_data['hidden_size']},"
        f" n_heads=4, action_dim={yml_data['action_dim']}  \n"
        f"**Best epoch metrics:** v_acc={v_acc:.4f}, v_racc={v_racc:.4f}  \n"
        f"**Inference:** local CPU, beam_width={args.beam_width}  \n"
        f"**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  \n"
        f"**Routes:** [`{out_yml}`](../{out_yml})  \n"
    )

    rows = []
    for r in yml_data["routes"]:
        n_routes = r["n_routes"]
        n_mol_solved = r["n_solved"]
        elapsed = r["elapsed_s"]
        frac = round(n_mol_solved / n_routes, 3) if n_routes > 0 else 0.0
        best_prob = 0.0
        if r.get("routes"):
            best_prob = r["routes"][0].get("trajectory_prob", 0.0)
        if r.get("error"):
            best_prob_str = "ERR"
        elif n_routes == 0:
            best_prob_str = "—"
        elif best_prob >= 0.0001:
            best_prob_str = f"{best_prob:.4f}"
        else:
            best_prob_str = "~0"
        rows.append((
            r["query_name"], r.get("pubchem_cid", ""),
            n_routes, n_mol_solved, f"{frac:.3f}", best_prob_str, f"{elapsed:.1f}",
        ))

    table = (
        "| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    table += "".join("| " + " | ".join(str(x) for x in row) + " |\n" for row in rows)

    summary = (
        f"\n## Summary\n\n"
        f"{n_solved} of {n_valid} valid molecules had ≥1 solved route (skipped: {n_skipped}).\n\n"
    )

    with open(out_md, "w") as f:
        f.write(header + summary + table + "\n")

    return out_md


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="results/hypertune-large-emma-24-26_layer/trial_000",
    )
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument(
        "--study", default="hypertune-large-emma-24-26_layer",
        help="Study name embedded in yml/md metadata",
    )
    parser.add_argument("--trial", default="trial_000")
    parser.add_argument(
        "--out-name", default="large_emma_24layers_trial000",
        help="Stem for output files (data/test_molecules_routes_<out-name>.yml etc.)",
    )
    parser.add_argument(
        "--molecules", default="data/test_molecules.yml",
        help="Path to test molecules YAML (default: data/test_molecules.yml)",
    )
    args = parser.parse_args()

    os.chdir(REPO)

    config_path = f"{args.model_dir}/model.config.yaml"
    if not os.path.exists(config_path):
        config_path = f"{args.model_dir}/config.yaml"
    config = read_config(config_path)
    model_path = f"{args.model_dir}/model.pth"

    best_epoch, best_metrics = _best_epoch_metrics(args.model_dir)

    print(f"Model : {model_path}", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print("Loading model...", flush=True)
    model = init_model(config, model_path=model_path)
    print("Model loaded.", flush=True)

    predictor = RoutePredictor(model, config, beam_width=args.beam_width)

    with open(args.molecules) as f:
        mol_data = yaml.safe_load(f)
    molecules = [m for m in mol_data["molecules"] if m.get("smiles") or m.get("canonical_smiles")]
    print(f"\nRunning inference on {len(molecules)} molecules (beam_width={args.beam_width})\n", flush=True)

    routes_output = []
    n_solved = 0
    n_skipped = 0
    total_start = time.time()

    for i, mol in enumerate(molecules, 1):
        smiles = mol.get("canonical_smiles") or mol.get("smiles")
        name = mol.get("query_name", f"mol_{i}")
        cid = mol.get("pubchem_cid", "")

        print(f"[{i}/{len(molecules)}] {name} (CID {cid}) ...", end=" ", flush=True)
        t0 = time.time()
        try:
            beams = predictor.predict_all_routes(
                smiles, beam_width=args.beam_width, target_reward=0.5
            )
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"ERROR: {exc}", flush=True)
            routes_output.append({
                "query_name": name,
                "pubchem_cid": cid,
                "smiles": smiles,
                "n_routes": 0,
                "n_solved": 0,
                "elapsed_s": round(elapsed, 3),
                "error": str(exc),
                "routes": [],
            })
            n_skipped += 1
            continue

        elapsed = time.time() - t0
        n_mol_solved = sum(1 for b in beams if b.route_solved)
        if n_mol_solved > 0:
            n_solved += 1

        print(f"{len(beams)} routes, {n_mol_solved} solved, {elapsed:.1f}s", flush=True)

        routes_output.append({
            "query_name": name,
            "pubchem_cid": cid,
            "smiles": smiles,
            "n_routes": len(beams),
            "n_solved": n_mol_solved,
            "elapsed_s": round(elapsed, 3),
            "routes": [_beam_to_route_dict(b) for b in beams],
        })

    total_elapsed = time.time() - total_start
    n_valid = len(molecules) - n_skipped

    yml_data = {
        "model": "large",
        "study": args.study,
        "trial": args.trial,
        "model_path": model_path,
        "n_layers": config["model"]["n_layers"],
        "hidden_size": config["model"]["hidden_size"],
        "action_dim": config["dataset"]["action_dim"],
        "best_epoch": best_epoch,
        "best_valid_route_accuracy": best_metrics.get("valid_route_accuracy"),
        "best_valid_action_accuracy": best_metrics.get("valid_action_accuracy"),
        "beam_width": args.beam_width,
        "n_molecules": len(molecules),
        "n_solved": n_solved,
        "n_skipped": n_skipped,
        "frac_solved": round(n_solved / n_valid, 3) if n_valid > 0 else 0.0,
        "total_elapsed_s": round(total_elapsed, 1),
        "date": str(date.today()),
        "routes": routes_output,
    }

    out_yml = f"data/test_molecules_routes_{args.out_name}.yml"
    with open(out_yml, "w") as f:
        yaml.dump(yml_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"\nSaved routes: {out_yml}", flush=True)

    out_md = _generate_md(yml_data, args, out_yml, best_epoch, best_metrics)
    print(f"Saved report: {out_md}", flush=True)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {n_solved}/{n_valid} solved ({n_solved/n_valid:.1%})  total {total_elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
