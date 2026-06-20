# RetroSynFormer — hypertune-large-nonuniform-dropout-12-layers / trial_000

**Date:** 2026-06-19  
**Study:** `hypertune-large-nonuniform-dropout-12-layers`  
**Trial:** `trial_000` — best checkpoint epoch 91 (min valid_loss=0.04743)  
**Architecture:** 12 layers, hidden_size=640, n_heads=5, action_dim=2957  
**Best epoch metrics:** v_acc=0.3133, v_racc=0.0637  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_large_nonuniform_dropout_12_layers_trial_000.yml`](../data/test_molecules_routes_large_nonuniform_dropout_12_layers_trial_000.yml)

## Summary

8 of 23 valid molecules had ≥1 solved route (skipped: 2).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Methoxy_Diphenylamine | 11435828 | 199 | 131 | 0.658 | 0.0078 | 8.2 |
| Tolyl_Pyridine | 603589 | 186 | 100 | 0.538 | 0.0058 | 8.5 |
| Imatinib | 5291 | 145 | 72 | 0.497 | 0.0027 | 9.0 |
| Etoricoxib | 123619 | 154 | 56 | 0.364 | 0.0108 | 10.0 |
| Fluorinated_Imidazole | 84117446 | 120 | 34 | 0.283 | 0.0003 | 8.6 |
| Similar to Rivaroxaban | 9875401 | 99 | 2 | 0.020 | 0.0000 | 9.0 |
| Ibuprofen | 3672 | 91 | 1 | 0.011 | 0.0000 | 8.8 |
| Paclitaxel | 36314 | 91 | 0 | 0.000 | 0.0000 | 10.5 |
| Camlipixant | 76955630 | 98 | 0 | 0.000 | 0.0000 | 8.9 |
| Fluorinated_Imidazole | 56842878 | 98 | 0 | 0.000 | ~0 | 8.7 |
| Orforglipron | 137319706 | 95 | 0 | 0.000 | ~0 | 10.1 |
| Acalabrutinib | 71226662 | 85 | 0 | 0.000 | 0.0000 | 9.8 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | 0.0000 | 9.5 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 10.1 |
| Ozempic | 56843331 | 100 | 0 | 0.000 | ~0 | 14.6 |
| Apixaban | 10182969 | 100 | 0 | 0.000 | 0.0000 | 10.2 |
| Palbocyclib | 5330286 | 190 | 0 | 0.000 | ~0 | 15.3 |
| Etoposide | 36462 | 100 | 0 | 0.000 | 0.0000 | 10.9 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | 0.0000 | 10.2 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 100 | 0 | 0.000 | 0.0000 | 10.1 |
| losartan | 3961 | 100 | 0 | 0.000 | ~0 | 10.2 |
| Venetoclax | 49846579 | 100 | 0 | 0.000 | 0.0000 | 10.2 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with standard model (beam_width=10)

| Molecule | Standard frac_solved | This model frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Methoxy_Diphenylamine | 0.685 | 0.658 | -0.027 |
| Tolyl_Pyridine | 0.563 | 0.538 | -0.025 |
| Imatinib | 0.361 | 0.497 | +0.136 |
| Etoricoxib | 0.701 | 0.364 | -0.337 |
| Fluorinated_Imidazole | 0.000 | 0.283 | +0.283 |
| Similar to Rivaroxaban | 0.057 | 0.020 | -0.036 |
| Ibuprofen | 0.000 | 0.011 | +0.011 |
| Paclitaxel | 0.000 | 0.000 | = |
| Camlipixant | 0.000 | 0.000 | = |
| Fluorinated_Imidazole | 0.000 | 0.000 | = |
| Orforglipron | 0.000 | 0.000 | = |
| Acalabrutinib | 0.007 | 0.000 | -0.007 |
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
- Model trained for 92 epochs; checkpoint is best valid_loss epoch.
- Inference ran on local CPU; the 1.3 GB model loaded from `model.pth`.
