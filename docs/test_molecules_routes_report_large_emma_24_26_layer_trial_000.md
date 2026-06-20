# RetroSynFormer — hypertune-large-emma-24-26_layer / trial_000

**Date:** 2026-06-19  
**Study:** `hypertune-large-emma-24-26_layer`  
**Trial:** `trial_000` — best checkpoint epoch 1 (min valid_loss=0.16760)  
**Architecture:** 24 layers, hidden_size=1024, n_heads=4, action_dim=2957  
**Best epoch metrics:** v_acc=0.2951, v_racc=0.0579  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_large_emma_24_26_layer_trial_000.yml`](../data/test_molecules_routes_large_emma_24_26_layer_trial_000.yml)

## Summary

9 of 23 valid molecules had ≥1 solved route (skipped: 2).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Methoxy_Diphenylamine | 11435828 | 227 | 163 | 0.718 | 0.0460 | 14.1 |
| Tolyl_Pyridine | 603589 | 354 | 197 | 0.556 | 0.0133 | 21.4 |
| Etoricoxib | 123619 | 142 | 62 | 0.437 | 0.0027 | 14.4 |
| Imatinib | 5291 | 161 | 67 | 0.416 | 0.0000 | 13.9 |
| Fluorinated_Imidazole | 84117446 | 111 | 29 | 0.261 | 0.0000 | 13.4 |
| Similar to Rivaroxaban | 9875401 | 104 | 4 | 0.038 | 0.0003 | 14.1 |
| Ibuprofen | 3672 | 86 | 1 | 0.012 | 0.0000 | 12.3 |
| Acalabrutinib | 71226662 | 91 | 1 | 0.011 | 0.0000 | 14.8 |
| Paclitaxel | 36314 | 100 | 0 | 0.000 | ~0 | 15.4 |
| Camlipixant | 76955630 | 93 | 0 | 0.000 | ~0 | 13.9 |
| Fluorinated_Imidazole | 56842878 | 95 | 0 | 0.000 | 0.0000 | 13.7 |
| Orforglipron | 137319706 | 97 | 0 | 0.000 | ~0 | 14.7 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | ~0 | 13.9 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 14.8 |
| Ozempic | 56843331 | 98 | 0 | 0.000 | 0.0000 | 18.7 |
| Apixaban | 10182969 | 97 | 0 | 0.000 | ~0 | 15.0 |
| Palbocyclib | 5330286 | 100 | 0 | 0.000 | ~0 | 14.2 |
| Etoposide | 36462 | 100 | 0 | 0.000 | ~0 | 15.1 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | ~0 | 15.7 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 100 | 0 | 0.000 | 0.0000 | 15.0 |
| losartan | 3961 | 100 | 0 | 0.000 | ~0 | 15.3 |
| Venetoclax | 49846579 | 100 | 0 | 0.000 | ~0 | 15.5 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES | — | — |
| Cefepime | 49846579 | — | — | no SMILES | — | — |

## Comparison with standard model (beam_width=10)

| Molecule | Standard frac_solved | This model frac_solved | Δ |
|---|---|---|---|
| Aspirin | 1.000 | 1.000 | = |
| Methoxy_Diphenylamine | 0.685 | 0.718 | +0.033 |
| Tolyl_Pyridine | 0.563 | 0.556 | -0.006 |
| Etoricoxib | 0.701 | 0.437 | -0.264 |
| Imatinib | 0.361 | 0.416 | +0.056 |
| Fluorinated_Imidazole | 0.000 | 0.261 | +0.261 |
| Similar to Rivaroxaban | 0.057 | 0.038 | -0.018 |
| Ibuprofen | 0.000 | 0.012 | +0.012 |
| Acalabrutinib | 0.007 | 0.011 | +0.004 |
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
- Model trained for 2 epochs; checkpoint is best valid_loss epoch.
- Inference ran on local CPU; the 1.3 GB model loaded from `model.pth`.
