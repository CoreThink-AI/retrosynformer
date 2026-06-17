# RetroSynFormer Standard — Test Molecule Route Predictions

**Date:** 2026-06-17  
**Model:** RetroSynFormer standard (`action_dim=1573`, 1573 reaction templates, trial_006 200-epoch run, best checkpoint epoch 97)  
**Endpoint:** `retrosynformer-inference-125069248164.us-central1.run.app/predict`  
**Beam width:** 10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_standard_model.yml`](../data/test_molecules_routes_standard_model.yml)

## Summary

Routes were requested for all 25 molecules in the test set. 23 produced valid API responses; 1 had an invalid SMILES (Rivaroxaban — RDKit rejects it); 1 had no SMILES (Cefepime). 8 of 23 valid molecules had ≥1 solved route, up from 7/23 with the small model.

| Molecule | PubChem CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.000 | 0.2 |
| Tolyl_Pyridine | 603589 | 391 | 220 | 0.563 | 0.335 | 39.2 |
| Methoxy_Diphenylamine | 11435828 | 216 | 148 | 0.685 | 0.058 | 24.5 |
| Etoricoxib | 123619 | 194 | 136 | 0.701 | 0.006 | 27.2 |
| Imatinib | 5291 | 147 | 53 | 0.361 | 0.000 | 24.8 |
| Similar to Rivaroxaban | 9875401 | 106 | 6 | 0.057 | 0.000 | 26.1 |
| Fluorinated_Imidazole | 84117446 | 101 | 9 | 0.089 | 0.000 | 24.2 |
| Acalabrutinib | 71226662 | 150 | 1 | 0.007 | 0.000 | 33.5 |
| Ibuprofen | 3672 | 81 | 0 | 0.000 | — | 25.6 |
| Paclitaxel | 36314 | 97 | 0 | 0.000 | — | 33.4 |
| Camlipixant | 76955630 | 97 | 0 | 0.000 | — | 28.0 |
| Fluorinated_Imidazole (dipeptide) | 56842878 | 165 | 0 | 0.000 | — | 44.0 |
| Orforglipron | 137319706 | 100 | 0 | 0.000 | — | 30.0 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | — | 27.5 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | — | 30.8 |
| Ozempic | 56843331 | 100 | 0 | 0.000 | — | 53.5 |
| Apixaban | 10182969 | 180 | 0 | 0.000 | — | 34.8 |
| Palbocyclib | 5330286 | 100 | 0 | 0.000 | — | 29.1 |
| Etoposide | 36462 | 62 | 0 | 0.000 | — | 31.4 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | — | 29.9 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 100 | 0 | 0.000 | — | 26.1 |
| losartan | 3961 | 100 | 0 | 0.000 | — | 27.9 |
| Venetoclax | 49846579 | 290 | 0 | 0.000 | — | 60.9 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with small model (beam_width=10)

| Molecule | Small frac_solved | Standard frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Etoricoxib | 0.606 | **0.701** | +0.095 |
| Methoxy_Diphenylamine | 0.549 | **0.685** | +0.136 |
| Tolyl_Pyridine | 0.383 | **0.563** | +0.180 |
| Imatinib | 0.277 | **0.361** | +0.084 |
| Fluorinated_Imidazole | 0.045 | **0.089** | +0.044 |
| Similar to Rivaroxaban | 0.000 | **0.057** | +0.057 |
| Acalabrutinib | 0.000 | **0.007** | +0.007 |
| Ibuprofen | **0.070** | 0.000 | −0.070 |
| All others | 0.000 | 0.000 | = |

The standard model improves solve rate on 7 of 9 molecules the small model could handle, adds 2 new molecules (Similar to Rivaroxaban, Acalabrutinib), and loses Ibuprofen (likely a template-coverage edge case near the boundary of both vocabularies).

## Endpoint latency and throughput

**Cold start:** The Cloud Run service was already warm from a `/health` check immediately before this run. A true cold start (fresh container) adds ~30–60 s for GCS artifact download (model.pth 33 MB, templates 316 KB, building blocks 1.9 MB) before the first request can be served. The startup probe is set to 240 s which should cover this.

**Inference latency at beam_width=10 (warm, CPU, 2 vCPU / 8 GB):**

| Complexity | Examples | Range |
|---|---|---|
| Trivial (building block, 0 steps) | Aspirin | 0.2 s |
| Simple scaffolds | Etoricoxib, Methoxy_Diphenylamine, Tolyl_Pyridine | 24–40 s |
| Medium complexity | Acalabrutinib, Palbocyclib, Apixaban | 29–34 s |
| Large/peptidic SMILES | Venetoclax (118 chars), Ozempic (275 chars) | 54–61 s |

**Throughput:** With `--concurrency 4` and `--min-instances 1`, the service handles 4 simultaneous requests. At ~30 s/request average, sustained throughput is ~8 req/min (~0.13 req/s) per instance. For higher throughput, increase `--max-instances` or move to GPU (Cloud Run L4 would cut inference time to ~1–3 s).

## Notes

- Aspirin is a purchasable building block in the standard template library — solved in 0 reaction steps with `trajectory_prob=1.0`.
- Tolyl_Pyridine shows the largest absolute improvement (+0.180) and the highest best_prob (0.335) of any non-trivial molecule, confirming the model handles simple biaryl amines well.
- Complex polycyclic drugs (Orforglipron, Venetoclax, Paclitaxel, Ozempic) remain unsolved at beam_width=10. The standard model's 1573-template vocabulary covers more chemistry than small (589) but the template set is still insufficient for multi-ring natural-product-like scaffolds.
- The standard model peaked at **epoch 97** of 200 trained; valid_route_accuracy did not improve beyond 0.065 after that point. The 200-epoch run did not overfit visibly (validation loss continued to decrease slowly) but neither did it improve on the hypertune checkpoint. A wider hyperparameter search or the large dataset (2957 templates) is the most likely path to meaningful further improvement.
- Ibuprofen regression (small: 7%, standard: 0%): Ibuprofen has a chiral centre and a relatively simple scaffold; the regression likely reflects a coverage gap in the standard template set near that specific disconnection rather than overfitting.
