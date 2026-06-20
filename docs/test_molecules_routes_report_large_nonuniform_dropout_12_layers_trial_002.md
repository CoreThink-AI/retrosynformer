# RetroSynFormer — hypertune-large-nonuniform-dropout-12-layers / trial_002

**Date:** 2026-06-19  
**Study:** `hypertune-large-nonuniform-dropout-12-layers`  
**Trial:** `trial_002` — best checkpoint epoch 36 (min valid_loss=0.08130)  
**Architecture:** 12 layers, hidden_size=640, n_heads=5, action_dim=2957  
**Best epoch metrics:** v_acc=0.2898, v_racc=0.0581  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_large_nonuniform_dropout_12_layers_trial_002.yml`](../data/test_molecules_routes_large_nonuniform_dropout_12_layers_trial_002.yml)

## Summary

9 of 23 valid molecules had ≥1 solved route (skipped: 2).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Methoxy_Diphenylamine | 11435828 | 210 | 148 | 0.705 | 0.0242 | 8.6 |
| Tolyl_Pyridine | 603589 | 203 | 130 | 0.640 | 0.0119 | 9.1 |
| Imatinib | 5291 | 163 | 79 | 0.485 | 0.0017 | 8.7 |
| Etoricoxib | 123619 | 146 | 64 | 0.438 | 0.0261 | 9.6 |
| Fluorinated_Imidazole | 84117446 | 117 | 29 | 0.248 | 0.0001 | 8.5 |
| Ibuprofen | 3672 | 91 | 1 | 0.011 | 0.0000 | 9.7 |
| Acalabrutinib | 71226662 | 93 | 1 | 0.011 | 0.0000 | 10.1 |
| Similar to Rivaroxaban | 9875401 | 98 | 1 | 0.010 | 0.0000 | 9.6 |
| Paclitaxel | 36314 | 99 | 0 | 0.000 | 0.0000 | 10.8 |
| Camlipixant | 76955630 | 98 | 0 | 0.000 | 0.0000 | 9.9 |
| Fluorinated_Imidazole | 56842878 | 97 | 0 | 0.000 | 0.0000 | 10.4 |
| Orforglipron | 137319706 | 100 | 0 | 0.000 | ~0 | 10.4 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | 0.0000 | 9.6 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 10.5 |
| Ozempic | 56843331 | 95 | 0 | 0.000 | 0.0000 | 14.0 |
| Apixaban | 10182969 | 100 | 0 | 0.000 | 0.0000 | 9.5 |
| Palbocyclib | 5330286 | 100 | 0 | 0.000 | ~0 | 9.6 |
| Etoposide | 36462 | 100 | 0 | 0.000 | 0.0000 | 10.7 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | 0.0000 | 10.8 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 100 | 0 | 0.000 | 0.0000 | 9.8 |
| losartan | 3961 | 100 | 0 | 0.000 | ~0 | 9.3 |
| Venetoclax | 49846579 | 100 | 0 | 0.000 | ~0 | 11.2 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with standard model (beam_width=10)

| Molecule | Standard frac_solved | This model frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Methoxy_Diphenylamine | 0.685 | 0.705 | +0.020 |
| Tolyl_Pyridine | 0.563 | 0.640 | +0.078 |
| Imatinib | 0.361 | 0.485 | +0.124 |
| Etoricoxib | 0.701 | 0.438 | -0.263 |
| Fluorinated_Imidazole | 0.000 | 0.248 | +0.248 |
| Ibuprofen | 0.000 | 0.011 | +0.011 |
| Acalabrutinib | 0.007 | 0.011 | +0.004 |
| Similar to Rivaroxaban | 0.057 | 0.010 | -0.046 |
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
- Model trained for 37 epochs; checkpoint is best valid_loss epoch.
- Inference ran on local CPU; the 1.3 GB model loaded from `model.pth`.
