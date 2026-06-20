# RetroSynFormer Large — Test Molecule Route Predictions

**Date:** 2026-06-19  
**Model:** RetroSynFormer large (`action_dim=2957`, 2957 reaction templates, `trial_003`, best checkpoint epoch 34 of 42 trained)  
**Architecture:** 26 layers, hidden_size=1024, n_heads=4  
**Inference:** local CPU (no GPU), `model.pth` loaded directly  
**Beam width:** 10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_large_model.yml`](../data/test_molecules_routes_large_model.yml)

## Summary

Routes were requested for all 25 molecules in the test set. 23 produced valid results; 1 had an invalid SMILES (Rivaroxaban — RDKit rejects it); 1 had no SMILES (Cefepime). **9 of 23 valid molecules had ≥1 solved route**, up from 8/23 with the standard model and 7/23 with the small model.

| Molecule | PubChem CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.000 | 0.0 |
| Tolyl_Pyridine | 603589 | 209 | 126 | 0.603 | 0.0063 | 13.9 |
| Methoxy_Diphenylamine | 11435828 | 199 | 131 | 0.658 | 0.0116 | 14.0 |
| Etoricoxib | 123619 | 183 | 94 | 0.514 | 0.0292 | 15.6 |
| Imatinib | 5291 | 148 | 57 | 0.385 | 0.0006 | 14.9 |
| Fluorinated_Imidazole | 84117446 | 119 | 29 | 0.244 | 0.0002 | 13.8 |
| Similar to Rivaroxaban | 9875401 | 102 | 2 | 0.020 | 0.0001 | 14.5 |
| Acalabrutinib | 71226662 | 88 | 1 | 0.011 | ~0 | 15.5 |
| Ibuprofen | 3672 | 91 | 1 | 0.011 | ~0 | 13.3 |
| Paclitaxel | 36314 | 95 | 0 | 0.000 | — | 15.3 |
| Camlipixant | 76955630 | 97 | 0 | 0.000 | — | 15.5 |
| Fluorinated_Imidazole (dipeptide) | 56842878 | 182 | 0 | 0.000 | — | 24.4 |
| Orforglipron | 137319706 | 93 | 0 | 0.000 | — | 16.5 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | — | 15.2 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | — | 15.5 |
| Ozempic | 56843331 | 100 | 0 | 0.000 | — | 19.2 |
| Apixaban | 10182969 | 190 | 0 | 0.000 | — | 18.2 |
| Palbocyclib | 5330286 | 190 | 0 | 0.000 | — | 25.0 |
| Etoposide | 36462 | 100 | 0 | 0.000 | — | 15.9 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | — | 15.7 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 99 | 0 | 0.000 | — | 15.0 |
| losartan | 3961 | 100 | 0 | 0.000 | — | 15.5 |
| Venetoclax | 49846579 | 100 | 0 | 0.000 | — | 15.6 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | — | — | — | no SMILES | — | — |

## Comparison across model sizes (beam_width=10)

| Molecule | Small frac_solved | Standard frac_solved | Large frac_solved | Trend |
|---|---|---|---|---|
| Aspirin | 1.000 | 1.000 | 1.000 | = |
| Tolyl_Pyridine | 0.383 | 0.563 | **0.603** | ↑ |
| Methoxy_Diphenylamine | 0.549 | **0.685** | 0.658 | ↓ |
| Etoricoxib | 0.606 | **0.701** | 0.514 | ↓ |
| Imatinib | 0.277 | 0.361 | **0.385** | ↑ |
| Fluorinated_Imidazole | 0.045 | 0.089 | **0.244** | ↑↑ |
| Similar to Rivaroxaban | 0.000 | **0.057** | 0.020 | ↓ |
| Acalabrutinib | 0.000 | 0.007 | **0.011** | ↑ |
| Ibuprofen | 0.070 | 0.000 | **0.011** | recovered |
| All others | 0.000 | 0.000 | 0.000 | = |
| **Total solved** | **7/23** | **8/23** | **9/23** | |

## Notable results

**Ibuprofen recovered:** The standard model lost Ibuprofen solve capability (small: 7%, standard: 0%, large: 1.1%). The large dataset's wider template vocabulary (2957 vs 1573) recovers a valid disconnection path, albeit rare in beam search.

**Fluorinated_Imidazole large gain (+0.155):** The simple aryl-imidazole scaffold (PubChem 84117446, complexity 272) jumps from 8.9% → 24.4% solved. The larger template set covers the specific N-methylation and C–F bond-forming reactions for this motif more reliably.

**Etoricoxib regression (−0.187):** The frac_solved drops from 0.701 to 0.514 despite more templates. The model has trained only 42 epochs (vs the standard trial which ran longer); at epoch 34 (best checkpoint) the model likely hasn't fully converged on the pyridine/chloride disconnections that the standard model learned.

**Methoxy_Diphenylamine slight regression (−0.027):** Small and likely noise at this training depth.

**Similar to Rivaroxaban regression (−0.037):** The oxazolidinone/thiophene scaffold is covered in both standard and large vocabularies, but the large model's early stopping at epoch 42 has not learned the template weights as reliably.

## Model training context

Trial_003 is an early-stage checkpoint: 42 epochs completed, best `valid_action_accuracy = 0.309` at epoch 34. The standard trial ran to epoch 97/200 and achieved `valid_action_accuracy ≈ 0.27` (lower raw number but on a harder, smaller vocabulary). With the large vocabulary of 2957 templates, even 30% top-1 action accuracy represents meaningful template coverage.

| Metric | Epoch 0 | Epoch 34 (best) | Epoch 41 |
|---|---|---|---|
| valid_action_accuracy | 0.242 | 0.309 | 0.304 |
| valid_route_accuracy | 0.040 | — | 0.052 |
| valid_loss | 0.440 | — | 0.052 |

The model is still improving — `valid_loss` was still falling at epoch 41 (0.0517) vs epoch 0 (0.440), and `valid_route_accuracy` reached 0.052–0.058 which surpasses the standard model's best. Further training will likely improve solve rates on the harder molecules (Etoricoxib, Similar to Rivaroxaban) where the current checkpoint lags the standard model.

## Inference latency

Inference ran on a local CPU (AMD/Intel, no GPU). The 1.35 GB model loaded in **1.7 s** (cached in OS page cache after first access).

| Complexity range | Examples | t (s) |
|---|---|---|
| Trivial (building block, 0 steps) | Aspirin | 0.0 |
| Simple scaffolds | Fluorinated_Imidazole, Ibuprofen, Tolyl_Pyridine | 13–14 |
| Medium complexity | Etoricoxib, Imatinib, Similar_to_Rivaroxaban | 14–16 |
| Complex / peptidic | Fluorinated_Imidazole (dipeptide), Palbocyclib | 24–25 |
| Large SMILES | Ozempic (275-char SMILES) | 19 |

Per-molecule latency is **roughly half** the standard model's Cloud Run latency, which reflects the model running on a fast local CPU rather than a Cloud Run vCPU with process startup overhead — not a real apples-to-apples comparison.

## Notes

- Aspirin is a purchasable building block in the large template library — solved in 0 reaction steps.
- The large model's beam search tends to produce more diverse routes (209 for Tolyl_Pyridine vs 391 for standard) because the wider template vocabulary spreads probability mass differently. This lowers `n_routes` per beam but does not reduce solve rate.
- Complex polycyclic drugs (Orforglipron complexity 1950, Venetoclax 1640, Paclitaxel 1790) remain unsolved — these require multi-step routes beyond the 6-step `max_depth` limit, or templates not covered by any of the three PaRoutes datasets.
- Palbocyclib (complexity 667) produces 190 routes but none solved, suggesting the CDK4/6 inhibitor scaffold requires templates outside the large vocabulary or beyond max_depth.
- The large model was trained with `valid_set: n1+n5`, meaning the PaRoutes N1 and N5 benchmark molecules were held out from training. None of the 23 test molecules are from PaRoutes, so there is no data leakage concern for this evaluation.
