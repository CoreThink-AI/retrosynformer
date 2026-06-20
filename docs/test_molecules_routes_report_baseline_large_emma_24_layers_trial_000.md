# RetroSynFormer — hypertune-baseline-large-emma-24-layers / trial_000

**Date:** 2026-06-19  
**Study:** `hypertune-baseline-large-emma-24-layers`  
**Trial:** `trial_000` — best checkpoint epoch 10 (min valid_loss=0.05153)  
**Architecture:** 12 layers, hidden_size=512, n_heads=2, action_dim=2957  
**Best epoch metrics:** v_acc=0.1100, v_racc=0.0048  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_baseline_large_emma_24_layers_trial_000.yml`](../data/test_molecules_routes_baseline_large_emma_24_layers_trial_000.yml)

## Summary

7 of 23 valid molecules had ≥1 solved route (skipped: 2).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Tolyl_Pyridine | 603589 | 240 | 191 | 0.796 | 0.9390 | 8.6 |
| Methoxy_Diphenylamine | 11435828 | 229 | 163 | 0.712 | 0.8497 | 9.7 |
| Etoricoxib | 123619 | 221 | 77 | 0.348 | 0.0735 | 13.9 |
| Imatinib | 5291 | 275 | 66 | 0.240 | 0.0087 | 14.9 |
| Fluorinated_Imidazole | 84117446 | 105 | 24 | 0.229 | 0.0000 | 8.0 |
| Similar to Rivaroxaban | 9875401 | 102 | 2 | 0.020 | 0.0000 | 9.0 |
| Ibuprofen | 3672 | 85 | 0 | 0.000 | 0.0000 | 8.7 |
| Paclitaxel | 36314 | 98 | 0 | 0.000 | 0.0000 | 10.7 |
| Camlipixant | 76955630 | 90 | 0 | 0.000 | 0.0000 | 9.1 |
| Fluorinated_Imidazole | 56842878 | 98 | 0 | 0.000 | 0.0000 | 9.6 |
| Orforglipron | 137319706 | 100 | 0 | 0.000 | ~0 | 9.9 |
| Acalabrutinib | 71226662 | 94 | 0 | 0.000 | 0.0000 | 10.2 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | 0.0000 | 8.8 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 10.4 |
| Ozempic | 56843331 | 97 | 0 | 0.000 | 0.0000 | 13.9 |
| Apixaban | 10182969 | 160 | 0 | 0.000 | 0.0001 | 12.6 |
| Palbocyclib | 5330286 | 100 | 0 | 0.000 | ~0 | 11.2 |
| Etoposide | 36462 | 100 | 0 | 0.000 | 0.0000 | 10.7 |
| Mintedanib | 135423438 | 95 | 0 | 0.000 | 0.0000 | 9.9 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 99 | 0 | 0.000 | 0.0000 | 9.6 |
| losartan | 3961 | 100 | 0 | 0.000 | 0.0000 | 9.1 |
| Venetoclax | 49846579 | 140 | 0 | 0.000 | ~0 | 15.2 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with standard model (beam_width=10)

| Molecule | Standard frac_solved | This model frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Tolyl_Pyridine | 0.563 | 0.796 | +0.233 |
| Methoxy_Diphenylamine | 0.685 | 0.712 | +0.027 |
| Etoricoxib | 0.701 | 0.348 | -0.353 |
| Imatinib | 0.361 | 0.240 | -0.121 |
| Fluorinated_Imidazole | 0.000 | 0.229 | +0.229 |
| Similar to Rivaroxaban | 0.057 | 0.020 | -0.037 |
| Ibuprofen | 0.000 | 0.000 | = |
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
- Model trained for 11 epochs; checkpoint is best valid_loss epoch.
- Inference ran on local CPU; the 1.3 GB model loaded from `model.pth`.
