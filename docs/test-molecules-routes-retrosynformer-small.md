# RetroSynFormer Small — Test Molecule Route Predictions

**Date:** 2026-06-16  
**Model:** RetroSynFormer small (`action_dim=589`, 589 reaction templates)  
**Endpoint:** `retrosynformer-inference-knq67derjq-uc.a.run.app/predict`  
**Beam width:** 10  
**Input:** [`data/test_molecules.yml`](../data/test_molecules.yml)  
**Routes:** [`data/test_molecules_routes_from_retrosynformer_small.yml`](../data/test_molecules_routes_from_retrosynformer_small.yml)

## Summary

Routes were requested for all 25 molecules in the test set. 23 produced valid API responses; 1 had an invalid SMILES (Rivaroxaban — the SMILES in the input file fails RDKit validation); 1 had no SMILES at all (Cefepime).

| Molecule | PubChem CID | n_routes | n_solved | frac_solved |
|---|---|---|---|---|
| Aspirin | 2244 | 1 | 1 | 1.000 |
| Methoxy_Diphenylamine | 11435828 | 166 | 91 | 0.549 |
| Etoricoxib | 123619 | 188 | 114 | 0.606 |
| Tolyl_Pyridine | 603589 | 133 | 51 | 0.383 |
| Imatinib | 5291 | 119 | 33 | 0.277 |
| Ibuprofen | 3672 | 100 | 7 | 0.070 |
| Fluorinated_Imidazole | 84117446 | 88 | 4 | 0.045 |
| Acalabrutinib | 71226662 | 143 | 0 | 0.000 |
| Apixaban | 10182969 | 195 | 0 | 0.000 |
| Camlipixant | 76955630 | 79 | 0 | 0.000 |
| Etoposide | 36462 | 61 | 0 | 0.000 |
| Fluorinated_Imidazole (dipeptide) | 56842878 | 98 | 0 | 0.000 |
| Ibrutinib | 24821094 | 100 | 0 | 0.000 |
| losartan | 3961 | 98 | 0 | 0.000 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 9870771 | 101 | 0 | 0.000 |
| Mintedanib | 135423438 | 190 | 0 | 0.000 |
| Omeprazole | 4594 | 100 | 0 | 0.000 |
| Orforglipron | 137319706 | 98 | 0 | 0.000 |
| Ozempic | 56843331 | 99 | 0 | 0.000 |
| Palbocyclib | 5330286 | 100 | 0 | 0.000 |
| Paclitaxel | 36314 | 98 | 0 | 0.000 |
| Similar to Rivaroxaban | 9875401 | 100 | 0 | 0.000 |
| Venetoclax | 49846579 | 150 | 0 | 0.000 |
| Rivaroxaban | 9870771 | — | — | invalid SMILES |
| Cefepime | 49846579 | — | — | no SMILES |

## Notes

- Aspirin is already a purchasable building block in the small-model template library, so it is solved in 0 steps with probability 1.0.
- The small model (589 templates) has a narrow template vocabulary. Structurally complex molecules (Orforglipron, Venetoclax, Ozempic, Paclitaxel) return 0 solved routes at beam_width=10; the standard or large model with a wider beam would likely do better.
- Etoricoxib has the highest raw solve count (114/188 = 61%), followed by Methoxy_Diphenylamine (91/166 = 55%) and Tolyl_Pyridine (51/133 = 38%) — all relatively simple biaryl or heteroaryl scaffolds well-covered by the small template set.
- The Rivaroxaban SMILES in `test_molecules.yml` (`CCOC(=O)N1CCC[C@H]1...`) does not match the molecule's PubChem entry and is rejected by RDKit. The `canonical_smiles` field is missing for that entry.
