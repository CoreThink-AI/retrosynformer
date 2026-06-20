# RetroSynFormer — hypertune-baseline-large-emma-21-layers / trial_001

**Date:** 2026-06-19  
**Study:** `hypertune-baseline-large-emma-21-layers`  
**Trial:** `trial_001` — best checkpoint epoch 98 (min valid_loss=0.04291)  
**Architecture:** 16 layers, hidden_size=1024, n_heads=4, action_dim=2957  
**Best epoch metrics:** v_acc=0.2834, v_racc=0.0466  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_baseline_large_emma_21_layers_trial_001.yml`](../data/test_molecules_routes_baseline_large_emma_21_layers_trial_001.yml)

## Summary

8 of 23 valid molecules had ≥1 solved route (skipped: 2).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Tolyl_Pyridine | 603589 | 207 | 132 | 0.638 | 0.0076 | 11.3 |
| Methoxy_Diphenylamine | 11435828 | 190 | 110 | 0.579 | 0.0091 | 12.0 |
| Imatinib | 5291 | 144 | 60 | 0.417 | 0.0020 | 12.3 |
| Etoricoxib | 123619 | 147 | 61 | 0.415 | 0.0136 | 12.7 |
| Fluorinated_Imidazole | 84117446 | 115 | 31 | 0.270 | 0.0001 | 11.5 |
| Similar to Rivaroxaban | 9875401 | 100 | 2 | 0.020 | 0.0000 | 12.5 |
| Ibuprofen | 3672 | 91 | 1 | 0.011 | 0.0000 | 11.2 |
| Paclitaxel | 36314 | 94 | 0 | 0.000 | 0.0000 | 13.7 |
| Camlipixant | 76955630 | 96 | 0 | 0.000 | 0.0000 | 13.2 |
| Fluorinated_Imidazole | 56842878 | 172 | 0 | 0.000 | ~0 | 15.2 |
| Orforglipron | 137319706 | 99 | 0 | 0.000 | ~0 | 13.9 |
| Acalabrutinib | 71226662 | 84 | 0 | 0.000 | 0.0000 | 12.7 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | 0.0000 | 12.3 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 12.9 |
| Ozempic | 56843331 | 93 | 0 | 0.000 | 0.0000 | 15.9 |
| Apixaban | 10182969 | 230 | 0 | 0.000 | 0.0000 | 17.8 |
| Palbocyclib | 5330286 | 190 | 0 | 0.000 | ~0 | 20.7 |
| Etoposide | 36462 | 100 | 0 | 0.000 | 0.0000 | 13.4 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | 0.0000 | 13.6 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 100 | 0 | 0.000 | 0.0000 | 12.4 |
| losartan | 3961 | 100 | 0 | 0.000 | ~0 | 12.4 |
| Venetoclax | 49846579 | 190 | 0 | 0.000 | 0.0000 | 21.4 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with standard model (beam_width=10)

| Molecule | Standard frac_solved | This model frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Tolyl_Pyridine | 0.563 | 0.638 | +0.075 |
| Methoxy_Diphenylamine | 0.685 | 0.579 | -0.106 |
| Imatinib | 0.361 | 0.417 | +0.056 |
| Etoricoxib | 0.701 | 0.415 | -0.286 |
| Fluorinated_Imidazole | 0.000 | 0.270 | +0.270 |
| Similar to Rivaroxaban | 0.057 | 0.020 | -0.037 |
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
- Model trained for 99 epochs; checkpoint is best valid_loss epoch.
- Inference ran on local CPU; the 1.3 GB model loaded from `model.pth`.
