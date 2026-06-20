# RetroSynFormer — hypertune-baseline-large-emma-21-layers / trial_003

**Date:** 2026-06-19  
**Study:** `hypertune-baseline-large-emma-21-layers`  
**Trial:** `trial_003` — best checkpoint epoch 59 (min valid_loss=0.04677)  
**Architecture:** 26 layers, hidden_size=1024, n_heads=4, action_dim=2957  
**Best epoch metrics:** v_acc=0.3044, v_racc=0.0499  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_baseline_large_emma_21_layers_trial_003.yml`](../data/test_molecules_routes_baseline_large_emma_21_layers_trial_003.yml)

## Summary

9 of 23 valid molecules had ≥1 solved route (skipped: 2).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Methoxy_Diphenylamine | 11435828 | 207 | 145 | 0.700 | 0.0116 | 14.3 |
| Tolyl_Pyridine | 603589 | 195 | 115 | 0.590 | 0.0063 | 13.4 |
| Etoricoxib | 123619 | 132 | 53 | 0.402 | 0.0034 | 13.4 |
| Imatinib | 5291 | 148 | 57 | 0.385 | 0.0006 | 14.7 |
| Fluorinated_Imidazole | 84117446 | 119 | 29 | 0.244 | 0.0002 | 13.7 |
| Similar to Rivaroxaban | 9875401 | 98 | 2 | 0.020 | 0.0001 | 14.9 |
| Acalabrutinib | 71226662 | 88 | 1 | 0.011 | 0.0000 | 15.4 |
| Ibuprofen | 3672 | 91 | 1 | 0.011 | 0.0000 | 14.4 |
| Paclitaxel | 36314 | 99 | 0 | 0.000 | ~0 | 15.3 |
| Camlipixant | 76955630 | 98 | 0 | 0.000 | 0.0000 | 14.5 |
| Fluorinated_Imidazole | 56842878 | 100 | 0 | 0.000 | ~0 | 13.8 |
| Orforglipron | 137319706 | 93 | 0 | 0.000 | ~0 | 16.3 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | 0.0000 | 15.5 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 15.4 |
| Ozempic | 56843331 | 99 | 0 | 0.000 | 0.0000 | 19.4 |
| Apixaban | 10182969 | 100 | 0 | 0.000 | 0.0000 | 16.5 |
| Palbocyclib | 5330286 | 190 | 0 | 0.000 | ~0 | 24.3 |
| Etoposide | 36462 | 100 | 0 | 0.000 | 0.0000 | 15.3 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | ~0 | 16.4 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 100 | 0 | 0.000 | 0.0000 | 15.6 |
| losartan | 3961 | 100 | 0 | 0.000 | ~0 | 14.6 |
| Venetoclax | 49846579 | 100 | 0 | 0.000 | ~0 | 14.5 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with standard model (beam_width=10)

| Molecule | Standard frac_solved | This model frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Methoxy_Diphenylamine | 0.685 | 0.700 | +0.015 |
| Tolyl_Pyridine | 0.563 | 0.590 | +0.027 |
| Etoricoxib | 0.701 | 0.402 | -0.300 |
| Imatinib | 0.361 | 0.385 | +0.025 |
| Fluorinated_Imidazole | 0.000 | 0.244 | +0.244 |
| Similar to Rivaroxaban | 0.057 | 0.020 | -0.036 |
| Acalabrutinib | 0.007 | 0.011 | +0.005 |
| Ibuprofen | 0.000 | 0.011 | +0.011 |
| Paclitaxel | 0.000 | 0.000 | = |
| Camlipixant | 0.000 | 0.000 | = |
| Fluorinated_Imidazole | 0.000 | 0.000 | = |
| Orforglipron | 0.000 | 0.000 | = |
| Ibrutinib | 0.000 | 0.000 | = |
| Omeprazole | 0.000 | 0.000 | = |
| Ozempic | 0.000 | 0.000 | = |
| Apixaban | 0.000 | 0.000 | = |
| Palbocyclib | 0.000 | 0.000 | = |
| Etoposide | 0.000 | 0.000 | = |
| Mintedanib | 0.000 | 0.000 | = |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 0.000 | 0.000 | = |
| losartan | 0.000 | 0.000 | = |
| Venetoclax | 0.000 | 0.000 | = |

## Notes

- action_dim=2957 (large dataset, 2957 templates)
- Model trained for 60 epochs; checkpoint is best valid_loss epoch.
- Inference ran on local CPU; the 1.3 GB model loaded from `model.pth`.
