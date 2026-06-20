# RetroSynFormer — hypertune-baseline-large-emma-21-layers / trial_000

**Date:** 2026-06-19  
**Study:** `hypertune-baseline-large-emma-21-layers`  
**Trial:** `trial_000` — best checkpoint epoch 99 (min valid_loss=0.05349)  
**Architecture:** 12 layers, hidden_size=1024, n_heads=4, action_dim=2957  
**Best epoch metrics:** v_acc=0.2669, v_racc=0.0429  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_baseline_large_emma_21_layers_trial_000.yml`](../data/test_molecules_routes_baseline_large_emma_21_layers_trial_000.yml)

## Summary

9 of 23 valid molecules had ≥1 solved route (skipped: 2).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Methoxy_Diphenylamine | 11435828 | 232 | 162 | 0.698 | 0.0078 | 10.9 |
| Etoricoxib | 123619 | 175 | 69 | 0.394 | 0.0079 | 13.2 |
| Tolyl_Pyridine | 603589 | 279 | 98 | 0.351 | 0.0117 | 14.9 |
| Imatinib | 5291 | 140 | 49 | 0.350 | 0.0020 | 12.1 |
| Fluorinated_Imidazole | 84117446 | 118 | 36 | 0.305 | 0.0001 | 10.3 |
| Similar to Rivaroxaban | 9875401 | 99 | 2 | 0.020 | 0.0001 | 12.0 |
| Ibuprofen | 3672 | 90 | 1 | 0.011 | 0.0000 | 10.4 |
| Acalabrutinib | 71226662 | 134 | 1 | 0.007 | 0.0000 | 14.3 |
| Paclitaxel | 36314 | 97 | 0 | 0.000 | ~0 | 13.2 |
| Camlipixant | 76955630 | 91 | 0 | 0.000 | 0.0000 | 13.1 |
| Fluorinated_Imidazole | 56842878 | 170 | 0 | 0.000 | 0.0000 | 12.7 |
| Orforglipron | 137319706 | 190 | 0 | 0.000 | ~0 | 17.2 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | 0.0000 | 11.8 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 11.6 |
| Ozempic | 56843331 | 100 | 0 | 0.000 | 0.0000 | 14.7 |
| Apixaban | 10182969 | 170 | 0 | 0.000 | 0.0000 | 15.3 |
| Palbocyclib | 5330286 | 100 | 0 | 0.000 | ~0 | 11.4 |
| Etoposide | 36462 | 100 | 0 | 0.000 | 0.0000 | 12.8 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | 0.0000 | 11.1 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 95 | 0 | 0.000 | 0.0000 | 12.2 |
| losartan | 3961 | 100 | 0 | 0.000 | ~0 | 11.7 |
| Venetoclax | 49846579 | 100 | 0 | 0.000 | 0.0000 | 12.9 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with standard model (beam_width=10)

| Molecule | Standard frac_solved | This model frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Methoxy_Diphenylamine | 0.685 | 0.698 | +0.013 |
| Etoricoxib | 0.701 | 0.394 | -0.307 |
| Tolyl_Pyridine | 0.563 | 0.351 | -0.211 |
| Imatinib | 0.361 | 0.350 | -0.011 |
| Fluorinated_Imidazole | 0.000 | 0.305 | +0.305 |
| Similar to Rivaroxaban | 0.057 | 0.020 | -0.036 |
| Ibuprofen | 0.000 | 0.011 | +0.011 |
| Acalabrutinib | 0.007 | 0.007 | = |
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
- Model trained for 100 epochs; checkpoint is best valid_loss epoch.
- Inference ran on local CPU; the 1.3 GB model loaded from `model.pth`.
