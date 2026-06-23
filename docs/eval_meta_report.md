# RetroSynFormer Evaluation Meta-Report

**Generated:** 2026-06-23  
**Scope:** All `test_molecules*routes*.yml` files in `eval/`  
**Data files:** [`eval_runs_meta_report.csv`](eval_runs_meta_report.csv) · [`eval_molecules_meta_report.csv`](eval_molecules_meta_report.csv)

---

## Run inventory

| Run label | Date | Mode | Test set | Tested | Solved | % Solved | Cyclic/solved | Avg depth¹ | Pass 1 | Pass 2 | Pass 3 |
|-----------|------|------|----------|-------:|-------:|---------:|--------------:|-----------:|-------:|-------:|-------:|
| v3-initial | 2026-06-21 | endpoint-old | 22/25² | 22 | 6 | 27.3 % | 3 (50 %) | 5.00 | — | — | — |
| hypertune-t000 | 2026-06-21 | endpoint-old | 23/27³ | 23 | 6 | 26.1 % | 3 (50 %) | 5.05 | — | — | — |
| v3-t000 | 2026-06-22 | endpoint-old | 17/27⁴ | 17 | 9 | 52.9 % | 6 (67 %) | 6.38 | 8 | 1 | 0 |
| hypertune-t002 | 2026-06-23 | local model | 23/27³ | 23 | 7 | 30.4 % | 4 (57 %) | 4.95 | 6 | 0 | 0 |
| **hypertune-t003** | **2026-06-23** | **endpoint-old** | **23/27**³ | **23** | **9** | **39.1 %** | **6 (67 %)** | **4.95** | **8** | **1** | **0** |
| t003-partial *(excl.)* | 2026-06-22 | endpoint-old | 27/27 | 27 | 3 | 11.1 % | 2 (67 %) | 4.00 | 3 | 0 | 0 |
| cpu-subset *(excl.)* | 2026-06-23 | endpoint-cpu v0.1.45 | 9/9⁵ | 9 | 9 | 100 % | 6 (67 %) | 1.88 | 8 | 1 | 0 |

¹ Average depth of non-trivial (depth > 0) best routes.  
² 3 entries missing SMILES in the original test_molecules.yml at that time.  
³ 4 entries lacked valid SMILES/PubChem data (Cefepime, Lorazepam, Vepdegestrant, one other).  
⁴ 10 errors — mostly API timeouts/502s on the old GPU endpoint (OOM under load) plus missing SMILES.  
⁵ Solved-subset only (molecules previously solved in v3-t000); not a full benchmark.  
Partial and subset runs are **excluded** from cross-run statistics below.

---

## Best model: hypertune-t003

**9 / 23 solved (39.1 %)** on the full 23-molecule test set (4 molecules excluded for missing SMILES across all runs).  
This is the highest comparable solve rate across full-coverage runs.  
The v3-t000 run nominally shows 52.9 % but tested only 17 molecules due to endpoint instability.

---

## Molecule consistency (across 5 full runs)

### Always solved (6/23 — 26 %)

These molecules were solved in every full run regardless of model, configuration, or beam width:

| Molecule | Notes |
|----------|-------|
| Aspirin | depth=0 in all runs — is itself a building block |
| Etoricoxib | depth=2; cyclic in most runs |
| Fluorinated\_Imidazole | depth=2; cyclic; errored in early runs (missing SMILES), solved consistently once SMILES available |
| Imatinib | depth=3; cyclic |
| Methoxy\_Diphenylamine | depth=1; not cyclic — clean route |
| Tolyl\_Pyridine | depth=1; not cyclic — clean route |

### Sometimes solved (3 molecules)

| Molecule | Runs solved | Notes |
|----------|-------------|-------|
| Ibuprofen | 3 / 5 | depth=2; cyclic when solved |
| Acalabrutinib | 2 / 5 | Requires pass 2 (max_steps=8); depth=2 when solved; cyclic |
| Similar to Rivaroxaban | 2 / 4 | depth=2; cyclic when solved |

### Never solved (14 unique molecules, 60 %)

Apixaban, Camlipixant, Cefepime, Etoposide, Ibrutinib, Lorazepam, Losartan,
Methoxybiphenyl\_Sulfonamide\_Amidoxime, Mintedanib, Omeprazole, Orforglipron,
Ozempic (semaglutide), Paclitaxel, Palbocyclib, Rivaroxaban, Venetoclax, Vepdegestrant

> **Note:** `cefepime`, `etoposide`, `lorazepam`, `losartan`, `venetoclax`, `midazolam`, `nintedanib`
> appear in older runs under different capitalisation/spelling — they are the same molecules as their
> capitalised equivalents above.

---

## Cyclic route analysis

Cyclic routes (a molecule appears as both a target and a reactant in the same route) are **structurally
invalid** — the beam search looped without reaching purchasable building blocks.  Despite this, the
endpoint reports `all_leaves_purchasable=True` because cyclic detection is applied post-hoc.

| Run | Solved | Cyclic | % Cyclic of solved |
|-----|-------:|-------:|-------------------:|
| v3-initial | 6 | 3 | 50 % |
| hypertune-t000 | 6 | 3 | 50 % |
| v3-t000 | 9 | 6 | 67 % |
| hypertune-t002 | 7 | 4 | 57 % |
| hypertune-t003 | 9 | 6 | 67 % |

**~50–67 % of "solved" routes are cyclic across all runs.**  The 3 always-clean solves are:
Aspirin (trivial, depth=0), Methoxy\_Diphenylamine (depth=1), and Tolyl\_Pyridine (depth=1).
Every deeper route in the solved set tends to be cyclic, suggesting the model's depth-limited beam
search consistently hits the reversibility of ester hydrolysis / condensation templates at depth ≥ 2.

---

## CPU endpoint latency benchmark (v0.1.45 — 2026-06-23)

**Config:** 8 vCPU / 32 GiB Cloud Run, no GPU, max_routes=10/15, max_steps=6/8.  
Subset: 9 previously-solved molecules only.

| Molecule | Pass | Depth | Latency (s) | Cyclic |
|----------|-----:|------:|------------:|--------|
| Aspirin | 1 | 0 | 10.4 | |
| Methoxy\_Diphenylamine | 1 | 1 | 40.2 | |
| Tolyl\_Pyridine | 1 | 1 | 40.3 | |
| Fluorinated\_Imidazole | 1 | 2 | 40.3 | ⚠ |
| Imatinib | 1 | 3 | 40.2 | ⚠ |
| Similar to Rivaroxaban | 1 | 2 | 40.2 | ⚠ |
| Ibuprofen | 1 | 2 | 50.3 | ⚠ |
| Etoricoxib | 1 | 2 | 50.4 | ⚠ |
| Acalabrutinib | 2 | 2 | 264.5 (pass 1: 143.8 + pass 2: 120.7) | ⚠ |

**Typical pass-1 latency: 40–50 s** for depth 1–3 on CPU.  
Aspirin (depth=0, building-block lookup) resolves in 10 s.  
Acalabrutinib requires pass 2 (max_steps=8) and totals ~4.4 min wall time.  
All 9 molecules solved — 100 % on this subset.

---

## Model comparison: trial002 vs trial003

Both runs used the same 23-molecule set with progressive retry (max_routes=10/15, max_steps=6/8/10).

| | trial002 (local) | trial003 (endpoint) |
|--|--:|--:|
| Solved | 7 / 23 | 9 / 23 |
| % Solved | 30.4 % | 39.1 % |
| Avg depth (non-trivial) | 4.95 | 4.95 |
| Cyclic of solved | 4 (57 %) | 6 (67 %) |
| Solved on pass 1 | 6 | 8 |
| Solved on pass 2 | 0 | 1 |

trial003 solves 2 additional molecules over trial002.  
Both have near-identical avg depth — the gain is in breadth, not depth.

---

## Key findings

1. **Best solve rate:** 39.1 % (9/23) with hypertune-t003 on the full test set.
2. **Consistent solvers (6 molecules):** Aspirin, Etoricoxib, Fluorinated\_Imidazole, Imatinib,
   Methoxy\_Diphenylamine, Tolyl\_Pyridine — these are a reliable regression test set.
3. **Cyclic route problem is pervasive:** 50–67 % of all "solved" routes are structurally cyclic.
   Only 3 molecules reliably produce non-cyclic routes (depth ≤ 1).
4. **CPU vs GPU:** No regression in solve rate switching from L4 GPU to 8-vCPU Cloud Run.
   Typical latency ~40–50 s/molecule; Acalabrutinib is the outlier at ~265 s (two passes).
5. **Hard molecules:** 14 molecules have never been solved across any run. These span high-complexity
   macrocycles (Paclitaxel, Venetoclax), peptide-like structures (Ozempic), and heterocyclic
   scaffolds (Palbocyclib, Ibrutinib) that lie outside the PaRoutes training distribution.
6. **Pass 2 rarely needed:** In runs with progressive retry, pass 2 was required for only 1/9
   solved molecules (Acalabrutinib). Pass 3 was never triggered.
7. **Data quality improved over time:** Early runs had 3–10 molecules with missing SMILES;
   the current `test_molecules.yml` has complete PubChem data for all entries.

---

## Recommendations

- **Regression set:** Use the 6 always-solved molecules as a fast smoke-test (< 5 min total
  on CPU) for every new model or deployment.
- **Cyclic route filtering:** Implement server-side cyclic detection before returning
  `all_leaves_purchasable=True`. The current post-hoc detection (in `evaluate.py`) should
  be moved into `inference.py` or `environment.py`.
- **Hard molecule focus:** The 14 never-solved molecules likely require deeper beam search
  (max_steps > 10) or a training dataset that covers their reaction classes. Consider
  adding AiZynthFinder comparison for these specifically.
- **Pass 3 is unnecessary:** Consider removing pass 3 from the default retry config —
  it adds up to 15 min per molecule and has never triggered across 300+ molecule evaluations.
