# RetroSynFormer — hypertune-large-emma-24-26_layer / trial_000

**Date:** 2026-06-20  
**Study:** `hypertune-large-emma-24-26_layer`  
**Trial:** `trial_000` — best checkpoint epoch 24 (min valid_loss=0.05217)  
**Architecture:** 24 layers, hidden_size=1024, n_heads=4, action_dim=2957  
**Best epoch metrics:** v_acc=0.3444, v_racc=0.0705  
**Inference:** local CPU, beam_width=10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_large_emma_24layers_trial000.yml`](../data/test_molecules_routes_large_emma_24layers_trial000.yml)  

## Summary

9 of 23 valid molecules had ≥1 solved route (skipped: 1).

| Molecule | CID | n_routes | n_solved | frac_solved | best_prob | t (s) |
|---|---|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 | 1.0000 | 0.0 |
| Ibuprofen | 3672 | 91 | 1 | 0.011 | ~0 | 8.1 |
| Paclitaxel | 36314 | 98 | 0 | 0.000 | ~0 | 10.2 |
| Etoricoxib | 123619 | 161 | 59 | 0.366 | 0.0086 | 9.5 |
| Camlipixant | 76955630 | 98 | 0 | 0.000 | ~0 | 9.1 |
| Fluorinated_Imidazole | 84117446 | 111 | 36 | 0.324 | ~0 | 9.0 |
| Fluorinated_Imidazole | 56842878 | 167 | 0 | 0.000 | ~0 | 15.8 |
| Methoxy_Diphenylamine | 11435828 | 209 | 139 | 0.665 | 0.0114 | 10.0 |
| Tolyl_Pyridine | 603589 | 200 | 147 | 0.735 | 0.0072 | 8.5 |
| Orforglipron | 137319706 | 100 | 0 | 0.000 | ~0 | 10.4 |
| Acalabrutinib | 71226662 | 94 | 1 | 0.011 | ~0 | 9.3 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 | ~0 | 9.3 |
| Omeprazole | 4594 | 100 | 0 | 0.000 | ~0 | 9.7 |
| Imatinib | 5291 | 146 | 57 | 0.390 | 0.0004 | 9.1 |
| Ozempic | 56843331 | 190 | 0 | 0.000 | ~0 | 15.7 |
| Apixaban | 10182969 | 100 | 0 | 0.000 | ~0 | 9.7 |
| Palbocyclib | 5330286 | 100 | 0 | 0.000 | ~0 | 9.6 |
| Similar to Rivaroxaban | 9875401 | 103 | 3 | 0.029 | 0.0002 | 8.9 |
| Rivaroxaban | 9870771 | 0 | 0 | 0.000 | ERR | 0.0 |
| Etoposide | 36462 | 100 | 0 | 0.000 | ~0 | 10.7 |
| Mintedanib | 135423438 | 100 | 0 | 0.000 | ~0 | 10.7 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 260 | 0 | 0.000 | ~0 | 14.5 |
| losartan | 3961 | 100 | 0 | 0.000 | ~0 | 9.3 |
| Venetoclax | 49846579 | 120 | 0 | 0.000 | ~0 | 12.7 |

