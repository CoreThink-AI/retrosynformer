# Retrosynthesis Accuracy Report — trial_002 (GCP Endpoint)

**Date:** 2026-06-20  
**Endpoint:** `retrosynformer-inference-v3` Cloud Run  
**Model:** hypertune-large-emma-24-26_layer / trial_002, epoch 59  
**Eval params:** `max_steps=10`, `max_routes=10`  
**Test set:** `data/test_molecules.yml` (25 molecules)

---

## Summary

| Metric | Value |
|--------|-------|
| Molecules tested | 23 |
| Molecules solved | **9 / 23 (39%)** |
| Skipped (no SMILES) | 1 (Cefepime) |
| Errors (bad SMILES) | 1 (Rivaroxaban) |
| Avg time per molecule | 51.3 s |

---

## Comparison to Baselines

| Run | Solved | Notes |
|-----|--------|-------|
| 21L baseline, trial_000, bw=10, epoch=99 | 9/24 = **37%** | full set |
| 24L baseline, trial_000, bw=10, epoch=10 | 7/24 = **29%** | early stop, undertrained |
| **trial_002, epoch=59, s10/r10 (this run)** | 9/23 = **39%** | full set minus 1 error |
| 24-26L trial_000, bw=50 | 9/10 = 90% | 10-molecule subset only |

Trial_002 at epoch 59 slightly outperforms the 21-layer baseline (39% vs 37%), despite training not being complete (study still running). Direct comparison is approximate: the 21L baseline used the `/predict` endpoint schema; this run used `/retrosynthesis`.

---

## Per-Molecule Results

| Status | Molecule | Complexity | Routes | Solved | Depth | Best Score |
|--------|----------|-----------|--------|--------|-------|-----------|
| ✅ | Aspirin | 212 | 1 | 1 | 0 | 1.0000 |
| ✅ | Methoxy_Diphenylamine | 191 | 10 | 10 | 1 | 0.0149 |
| ✅ | Tolyl_Pyridine | 162 | 10 | 10 | 1 | 0.0082 |
| ✅ | Ibuprofen | 203 | 10 | 1 | 2 | 0.0000 |
| ✅ | Fluorinated_Imidazole (simple, c=272) | 272 | 10 | 10 | 2 | 0.0000 |
| ✅ | Etoricoxib | 514 | 10 | 10 | 2 | 0.0173 |
| ✅ | Acalabrutinib | 845 | 10 | 1 | 2 | 0.0000 |
| ✅ | Similar to Rivaroxaban | 589 | 10 | 3 | 2 | 0.0001 |
| ✅ | Imatinib | 742 | 10 | 10 | 4 | 0.0004 |
| ❌ | Omeprazole | 339 | 10 | 0 | 10 | — |
| ❌ | Apixaban | 582 | 10 | 0 | 10 | — |
| ❌ | losartan | 492 | 10 | 0 | 10 | — |
| ❌ | Fluorinated_Imidazole (complex, c=632) | 632 | 10 | 0 | 10 | — |
| ❌ | Palbocyclib | 667 | 10 | 0 | 10 | — |
| ❌ | Camlipixant | 704 | 10 | 0 | 10 | — |
| ❌ | Ibrutinib | 763 | 10 | 0 | 10 | — |
| ❌ | Mintedanib | 754 | 10 | 0 | 10 | — |
| ❌ | Methoxybiphenyl_Sulfonamide_Amidoxime | 843 | 10 | 0 | 10 | — |
| ❌ | Etoposide | 804 | 10 | 0 | 10 | — |
| ❌ | Paclitaxel | 1790 | 10 | 0 | 10 | — |
| ❌ | Orforglipron | 1950 | 10 | 0 | 10 | — |
| ❌ | Ozempic | 1700 | 10 | 0 | 10 | — |
| ❌ | Venetoclax | 1640 | 10 | 0 | 11 | — |
| ⚠️ | Rivaroxaban | 589 | — | — | — | invalid SMILES |
| ⏭️ | Cefepime | — | — | — | — | no SMILES |

---

## Observations

**Solved molecules** are predominantly simpler structures (complexity ≤ 514) found at shallow depth (1–2 steps), with two exceptions: Acalabrutinib (845) and Imatinib (742) were both solved despite high complexity. Imatinib was solved by all 10 beam routes at depth 4, suggesting the model has strong template coverage for its core scaffold.

**Unsolved at moderate complexity**: Omeprazole (339), losartan (492), and Apixaban (582) all failed despite relatively modest complexity scores. These may involve template gaps or unusual bond-forming reactions.

**Unsolved at high complexity**: The very high-complexity molecules (Paclitaxel 1790, Orforglipron 1950, Ozempic 1700, Venetoclax 1640) were all unsolved, as expected — multi-step synthesis of these macrolides and peptides is well beyond the 10-step depth limit.

**Duplicate Fluorinated_Imidazole**: The test set contains two molecules with this name — complexity 272 (solved, depth 2) and complexity 632 (unsolved, all 10 routes exhausted at depth 10). They are distinct SMILES.

**Rivaroxaban SMILES error**: The `canonical_smiles` field in `test_molecules.yml` for Rivaroxaban contains a stereocentre encoding rejected by the endpoint's RDKit validation. The `smiles` (isomeric) field was not tried. Fix: update the SMILES or use the isomeric field as fallback.

---

## Data

Full results: `data/test_molecules_routes_trial002_retrosynthesis_s10_r10.yml`
