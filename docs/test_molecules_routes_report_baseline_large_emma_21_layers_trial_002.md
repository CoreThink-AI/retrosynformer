# RetroSynFormer — hypertune-baseline-large-emma-21-layers / trial_002

**Date:** 2026-06-19  
**Study:** `hypertune-baseline-large-emma-21-layers`  
**Trial:** `trial_002` — best checkpoint epoch 98 (min valid_loss=0.05222)  
**Architecture:** 21 layers, hidden_size=1024, n_heads=4, action_dim=2957  
**Best epoch metrics:** v_acc=0.2937, v_racc=0.0510  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_baseline_large_emma_21_layers_trial_002.yml`](../data/test_molecules_routes_baseline_large_emma_21_layers_trial_002.yml)

## Summary

8 of 23 valid molecules had ≥1 solved route (skipped: 2).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Methoxy_Diphenylamine | 11435828 | 216 | 148 | 0.685 | 0.0112 | 12.0 |
| Tolyl_Pyridine | 603589 | 187 | 120 | 0.642 | 0.0077 | 13.4 |
| Imatinib | 5291 | 147 | 58 | 0.395 | 0.0006 | 12.3 |
| Etoricoxib | 123619 | 142 | 51 | 0.359 | 0.0043 | 17.4 |
| Fluorinated_Imidazole | 84117446 | 114 | 27 | 0.237 | 0.0001 | 11.8 |
| Similar to Rivaroxaban | 9875401 | 88 | 2 | 0.023 | 0.0003 | 14.6 |
| Ibuprofen | 3672 | 92 | 2 | 0.022 | 0.0000 | 12.7 |
| Paclitaxel | 36314 | 98 | 0 | 0.000 | 0.0000 | 14.8 |
| Camlipixant | 76955630 | 100 | 0 | 0.000 | 0.0000 | 14.3 |
| Fluorinated_Imidazole | 56842878 | 99 | 0 | 0.000 | 0.0000 | 12.9 |
| Orforglipron | 137319706 | 96 | 0 | 0.000 | ~0 | 14.8 |
| Acalabrutinib | 71226662 | 89 | 0 | 0.000 | 0.0000 | 13.6 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | 0.0000 | 14.6 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 13.9 |
| Ozempic | 56843331 | 96 | 0 | 0.000 | 0.0000 | 16.8 |
| Apixaban | 10182969 | 100 | 0 | 0.000 | 0.0000 | 14.6 |
| Palbocyclib | 5330286 | 100 | 0 | 0.000 | ~0 | 15.0 |
| Etoposide | 36462 | 100 | 0 | 0.000 | 0.0000 | 13.9 |
| Mintedanib | 135423438 | 98 | 0 | 0.000 | 0.0000 | 14.8 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 100 | 0 | 0.000 | 0.0000 | 13.7 |
| losartan | 3961 | 100 | 0 | 0.000 | ~0 | 14.6 |
| Venetoclax | 49846579 | 100 | 0 | 0.000 | 0.0000 | 14.2 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with standard model (beam_width=10)

| Molecule | Standard frac_solved | This model frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Methoxy_Diphenylamine | 0.685 | 0.685 | = |
| Tolyl_Pyridine | 0.563 | 0.642 | +0.079 |
| Imatinib | 0.361 | 0.395 | +0.034 |
| Etoricoxib | 0.701 | 0.359 | -0.342 |
| Fluorinated_Imidazole | 0.000 | 0.237 | +0.237 |
| Similar to Rivaroxaban | 0.057 | 0.023 | -0.034 |
| Ibuprofen | 0.000 | 0.022 | +0.022 |
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
