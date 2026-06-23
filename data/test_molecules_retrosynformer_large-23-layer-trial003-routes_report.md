# RetroSynFormer Evaluation Report

**Date:** 2026-06-23  
**Study:** large-23-layer  Trial: 003  
**Mode:** local model: /home/hobs/code/corethink/retrosynformer/results/hypertune-large-23-layer/trial_003/model.pth  
**Initial beam width (max_routes):** 10  
**Top routes saved:** 3  
**Progressive retry:** pass 1 (10/6), pass 2 (30/8), pass 3 (50/10)

---

## Summary

| Metric | Value |
|--------|-------|
| Molecules tested | 29 |
| Skipped (no SMILES / error) | 3 |
| **Solved** (all_leaves_purchasable) | **6/29** |
| Solved on pass 1 (max_routes=10, max_steps=6) | 5 |
| Solved on pass 2 (max_routes=15, max_steps=8) | 0 |
| Solved on pass 3 (max_routes=15, max_steps=10) | 0 |
| Trivially solved (depth=0, is a building block) | 1 |
| Cyclic best route | 25 |
| Avg depth of non-trivial best route | 5.0 |

---

## Per-Molecule Results

| Molecule | Complexity | Routes | Best depth | Solved | Pass | Cyclic | Leaves (purch/total) | Score |
|----------|-----------|--------|-----------|--------|------|--------|----------------------|-------|
| Aspirin | 212 | 1 | 0 | ✓ | 1 |  | 0/0 | 1.0000 |
| Ibuprofen | 203 | 3 | 1 | ✗ | — |  | 0/0 | 0.8823 |
| Paclitaxel | 1790 | 3 | 6 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Etoricoxib | 514 | 3 | 2 | ✓ | 1 | ⚠ | 3/3 | 0.0146 |
| Camlipixant | 704 | 3 | 6 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Fluorinated_Imidazole | 272 | 3 | 4 | ✗ | — | ⚠ | 2/2 | 0.0000 |
| Fluorinated_Imidazole | 632 | 3 | 6 | ✗ | — | ⚠ | 0/0 | 0.0000 |
| Methoxy_Diphenylamine | 191 | 3 | 1 | ✓ | 1 |  | 2/2 | 0.0120 |
| Tolyl_Pyridine | 162 | 3 | 1 | ✓ | 1 |  | 2/2 | 0.0195 |
| Orforglipron | 1950 | 3 | 4 | ✗ | — | ⚠ | 1/1 | 0.0000 |
| Acalabrutinib | 845 | 3 | 6 | ✗ | — | ⚠ | 4/4 | 0.0000 |
| Ibrutinib | 763 | 3 | 6 | ✗ | — | ⚠ | 4/4 | 0.0000 |
| Omeprazole | 339 | 3 | 6 | ✗ | — | ⚠ | 4/4 | 0.0000 |
| Imatinib | 742 | 3 | 4 | ✓ | 1 | ⚠ | 5/5 | 0.0000 |
| Ozempic | 1700 | 3 | 6 | ✗ | — | ⚠ | 0/0 | 0.0000 |
| Apixaban | 582 | 3 | 6 | ✗ | — | ⚠ | 4/4 | 0.0000 |
| Palbocyclib | 667 | 3 | 6 | ✗ | — | ⚠ | 2/2 | 0.0000 |
| Similar to Rivaroxaban | 645 | 3 | 6 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Rivaroxaban | 645 | 3 | 6 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Rivaroxaban | 589 | — | — | — | — | — | — | *invalid SMILES* |
| Etoposide | 804 | 3 | 6 | ✗ | — | ⚠ | 1/1 | 0.0000 |
| Similar to Mintedanib | 892 | 3 | 6 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Mintedanib | 754 | 3 | 6 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Methoxybiphenyl_Sulfonamide_Amidoxime | 843 | 3 | 6 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Similar to Losartan | 520 | 3 | 6 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Losartan | 492 | 3 | 5 | ✗ | — | ⚠ | 3/3 | 0.0000 |
| Venetoclax | 1640 | 3 | 6 | ✗ | — | ⚠ | 2/2 | 0.0000 |
| Similar to Cefepime | 869 | 3 | 5 | ✗ | — | ⚠ | 0/0 | 0.0000 |
| Cefepime | None | — | — | — | — | — | — | *no SMILES* |
| Similar to Lorazepam | 443 | 3 | 5 | ✗ | — | ⚠ | 1/1 | 0.0000 |
| Lorazepam | None | — | — | — | — | — | — | *no SMILES* |
| Vepdegestrant | 1310 | 3 | 6 | ✗ | — | ⚠ | 2/2 | 0.0000 |

---

## Per-Molecule Route Details

### Aspirin  (PubChem CID: 2244)
**SMILES:** `CC(=O)OC1=CC=CC=C1C(=O)O`  
**Best route:** depth=0  solved=True  score=1  

- *Target molecule is itself a building block (depth=0)*

### Ibuprofen  (PubChem CID: 3672)
**SMILES:** `CC(C)CC1=CC=C(C=C1)[C@@H](C)C(=O)O`  
**Best route:** depth=1  solved=False  score=0.8823  

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC(C)CC1=CC=C(C=C1)[C@@H](C)C(=O)O` → `COC(=O)[C@H](C)c1ccc(CC(C)C)cc1`

### Paclitaxel  (PubChem CID: 36314)
**SMILES:** `CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C` → `CO[C@]12C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)c3ccccc3)c3ccccc3)C(C)=C([C@@H](OC(C)=O)C(=O)[C@@]3(C)[C@H]([C@@H]1OC(=O)c1ccccc1)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]3O)C2(C)C`
2. `CO[C@]12C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)c3ccccc3)c3ccccc3)C(C)=C([C@@H](OC(C)=O)C(=O)[C@@]3(C)[C@H]([C@@H]1OC(=O)c1ccccc1)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]3O)C2(C)C` → `CO[C@]12C[C@H](OC(=O)[C@H](O)[C@@H](N)c3ccccc3)C(C)=C([C@@H](OC(C)=O)C(=O)[C@@]3(C)[C@H]([C@@H]1OC(=O)c1ccccc1)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]3O)C2(C)C` + `O=C(O)c1ccccc1`
3. `CO[C@]12C[C@H](OC(=O)[C@H](O)[C@@H](N)c3ccccc3)C(C)=C([C@@H](OC(C)=O)C(=O)[C@@]3(C)[C@H]([C@@H]1OC(=O)c1ccccc1)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]3O)C2(C)C` → `CO[C@]12C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)c3ccccc3)C(C)=C([C@@H](OC(C)=O)C(=O)[C@@]3(C)[C@H]([C@@H]1OC(=O)c1ccccc1)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]3O)C2(C)C`
4. `CO[C@]12C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)c3ccccc3)C(C)=C([C@@H](OC(C)=O)C(=O)[C@@]3(C)[C@H]([C@@H]1OC(=O)c1ccccc1)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]3O)C2(C)C` → `CC(=O)O[C@H]1C(=O)[C@@]2(C)[C@H]([C@H](OC(=O)c3ccccc3)[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)c4ccccc4)C(C)=C1C3(C)C)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]2O` + `CO`
5. `CC(=O)O[C@H]1C(=O)[C@@]2(C)[C@H]([C@H](OC(=O)c3ccccc3)[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)c4ccccc4)C(C)=C1C3(C)C)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]2O` → `CC(=O)O[C@H]1C(=O)[C@@]2(C)[C@H]([C@H](OC(=O)c3ccccc3)[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](N)c4ccccc4)C(C)=C1C3(C)C)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]2O` + `CC(C)(C)OC(=O)OC(=O)OC(C)(C)C`
6. `CC(=O)O[C@H]1C(=O)[C@@]2(C)[C@H]([C@H](OC(=O)c3ccccc3)[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](N)c4ccccc4)C(C)=C1C3(C)C)[C@]1(OC(C)=O)CO[C@@H]1C[C@@H]2O` → `CO[C@@H](C(=O)O[C@H]1C[C@@]2(O)[C@@H](OC(=O)c3ccccc3)[C@@H]3[C@]4(OC(C)=O)CO[C@@H]4C[C@H](O)[C@@]3(C)C(=O)[C@H](OC(C)=O)C(=C1C)C2(C)C)[C@@H](N)c1ccccc1`

**Building blocks proposed:**
- ✓ `O=C(O)c1ccccc1`
- ✓ `CO`
- ✓ `CC(C)(C)OC(=O)OC(=O)OC(C)(C)C`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Etoricoxib  (PubChem CID: 123619)
**SMILES:** `CC1=NC=C(C=C1)C2=C(C=C(C=N2)Cl)C3=CC=C(C=C3)S(=O)(=O)C`  
**Best route:** depth=2  solved=True  score=0.0146  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC1=NC=C(C=C1)C2=C(C=C(C=N2)Cl)C3=CC=C(C=C3)S(=O)(=O)C` → `CS(=O)(=O)c1ccc(-c2cc(Cl)cnc2Cl)cc1` + `Cc1ccc(B(O)O)cn1`
2. `CS(=O)(=O)c1ccc(-c2cc(Cl)cnc2Cl)cc1` → `CS(=O)(=O)c1ccc(B(O)O)cc1` + `Clc1cnc(Cl)c(Br)c1`

**Building blocks proposed:**
- ✓ `Cc1ccc(B(O)O)cn1`
- ✓ `CS(=O)(=O)c1ccc(B(O)O)cc1`
- ✓ `Clc1cnc(Cl)c(Br)c1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Camlipixant  (PubChem CID: 76955630)
**SMILES:** `CC1=CC2=NC(=C(N2C=C1)C[C@H]3CN(CCO3)C(=O)OC)C4=C(C=C(C=C4F)C(=O)NC)F`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC1=CC2=NC(=C(N2C=C1)C[C@H]3CN(CCO3)C(=O)OC)C4=C(C=C(C=C4F)C(=O)NC)F` → `CNC(=O)c1cc(F)c(-c2nc3cc(C)ccn3c2C[C@H]2CNCCO2)c(F)c1` + `COC(=O)Cl`
2. `CNC(=O)c1cc(F)c(-c2nc3cc(C)ccn3c2C[C@H]2CNCCO2)c(F)c1` → `Cc1ccn2c(C[C@H]3CNCCO3)c(-c3c(F)cc(C(=O)N(C)C(=O)OC(C)(C)C)cc3F)nc2c1`
3. `Cc1ccn2c(C[C@H]3CNCCO3)c(-c3c(F)cc(C(=O)N(C)C(=O)OC(C)(C)C)cc3F)nc2c1` → `Cc1ccn2c(C[C@H]3CN(C(=O)OCc4ccccc4)CCO3)c(-c3c(F)cc(C(=O)N(C)C(=O)OC(C)(C)C)cc3F)nc2c1`
4. `Cc1ccn2c(C[C@H]3CN(C(=O)OCc4ccccc4)CCO3)c(-c3c(F)cc(C(=O)N(C)C(=O)OC(C)(C)C)cc3F)nc2c1` → `CC(C)(C)OC(=O)OC(=O)OC(C)(C)C` + `CNC(=O)c1cc(F)c(-c2nc3cc(C)ccn3c2C[C@H]2CN(C(=O)OCc3ccccc3)CCO2)c(F)c1`
5. `CNC(=O)c1cc(F)c(-c2nc3cc(C)ccn3c2C[C@H]2CN(C(=O)OCc3ccccc3)CCO2)c(F)c1` → `Cc1ccn2c(C[C@H]3CN(C(=O)OCc4ccccc4)CCO3)c(-c3c(F)cc(C(=O)N(C)Cc4ccccc4)cc3F)nc2c1`
6. `Cc1ccn2c(C[C@H]3CN(C(=O)OCc4ccccc4)CCO3)c(-c3c(F)cc(C(=O)N(C)Cc4ccccc4)cc3F)nc2c1` → `Cc1ccn2c(C[C@H]3CNCCO3)c(-c3c(F)cc(C(=O)N(C)Cc4ccccc4)cc3F)nc2c1` + `O=C(Cl)OCc1ccccc1`

**Building blocks proposed:**
- ✓ `COC(=O)Cl`
- ✓ `CC(C)(C)OC(=O)OC(=O)OC(C)(C)C`
- ✓ `O=C(Cl)OCc1ccccc1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Fluorinated_Imidazole  (PubChem CID: 84117446)
**SMILES:** `CN1C=NC(=C1C2=CC=CC=C2F)C#N`  
**Best route:** depth=4  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CN1C=NC(=C1C2=CC=CC=C2F)C#N` → `Cn1cnc(C#N)c1Cl` + `OB(O)c1ccccc1F`
2. `Cn1cnc(C#N)c1Cl` → `Cn1cnc(C(N)=O)c1Cl`
3. `Cn1cnc(C(N)=O)c1Cl` → `Cn1cnc(C(=O)O)c1Cl` + `N`
4. `Cn1cnc(C(=O)O)c1Cl` → `COC(=O)c1ncn(C)c1Cl`

**Building blocks proposed:**
- ✓ `OB(O)c1ccccc1F`
- ✓ `N`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Fluorinated_Imidazole  (PubChem CID: 56842878)
**SMILES:** `C1C[C@H](N(C1)C(=O)[C@H](CC2=C(N=CN2)F)NC(=O)[C@@H]3CCC(=O)N3)C(=O)N`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C1C[C@H](N(C1)C(=O)[C@H](CC2=C(N=CN2)F)NC(=O)[C@@H]3CCC(=O)N3)C(=O)N` → `[N-]=[N+]=NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1[nH]cnc1F)NC(=O)[C@@H]1CCC(=O)N1`
2. `[N-]=[N+]=NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1[nH]cnc1F)NC(=O)[C@@H]1CCC(=O)N1` → `[N-]=[N+]=NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1[nH]cnc1F)NC(=O)[C@@H](N)CCC(=O)O`
3. `[N-]=[N+]=NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1[nH]cnc1F)NC(=O)[C@@H](N)CCC(=O)O` → `CCOC(=O)CC[C@H](N)C(=O)N[C@@H](Cc1[nH]cnc1F)C(=O)N1CCC[C@H]1C(=O)N=[N+]=[N-]`
4. `CCOC(=O)CC[C@H](N)C(=O)N[C@@H](Cc1[nH]cnc1F)C(=O)N1CCC[C@H]1C(=O)N=[N+]=[N-]` → `CCOC(=O)CC[C@H](N)C(=O)O` + `[N-]=[N+]=NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)Cc1[nH]cnc1F`
5. `[N-]=[N+]=NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)Cc1[nH]cnc1F` → `CC(C)(C)OC(=O)N[C@@H](Cc1[nH]cnc1F)C(=O)N1CCC[C@H]1C(=O)N=[N+]=[N-]`
6. `CC(C)(C)OC(=O)N[C@@H](Cc1[nH]cnc1F)C(=O)N1CCC[C@H]1C(=O)N=[N+]=[N-]` → `CC(C)(C)OC(=O)N[C@@H](Cc1[nH]cnc1F)C(=O)O` + `[N-]=[N+]=NC(=O)[C@@H]1CCCN1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Methoxy_Diphenylamine  (PubChem CID: 11435828)
**SMILES:** `CC1=CC=C(C=C1)NC2=CC=C(C=C2)OC`  
**Best route:** depth=1  solved=True  score=0.012  

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC1=CC=C(C=C1)NC2=CC=C(C=C2)OC` → `COc1ccc(N)cc1` + `Cc1ccc(Br)cc1`

**Building blocks proposed:**
- ✓ `COc1ccc(N)cc1`
- ✓ `Cc1ccc(Br)cc1`

### Tolyl_Pyridine  (PubChem CID: 603589)
**SMILES:** `CC1=CC=C(C=C1)NC2=CN=CC=C2`  
**Best route:** depth=1  solved=True  score=0.0195  

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC1=CC=C(C=C1)NC2=CN=CC=C2` → `Cc1ccc(Br)cc1` + `Nc1cccnc1`

**Building blocks proposed:**
- ✓ `Cc1ccc(Br)cc1`
- ✓ `Nc1cccnc1`

### Orforglipron  (PubChem CID: 137319706)
**SMILES:** `C[C@H]1C[C@]1(C2=NOC(=O)N2)N3C4=C(C=C(C=C4)[C@H]5CCOC(C5)(C)C)C=C3C(=O)N6CCC7=NN(C(=C7[C@@H]6C)N8C=CN(C8=O)C9=C(C1=C(C=C9)N(N=C1)C)F)C1=CC(=C(C(=C1)C)F)C`  
**Best route:** depth=4  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C[C@H]1C[C@]1(C2=NOC(=O)N2)N3C4=C(C=C(C=C4)[C@H]5CCOC(C5)(C)C)C=C3C(=O)N6CCC7=NN(C(=C7[C@@H]6C)N8C=CN(C8=O)C9=C(C1=C(C=C9)N(N=C1)C)F)C1=CC(=C(C(=C1)C)F)C` → `CI` + `Cc1cc(-n2nc3c(c2-n2ccn(-c4ccc5[nH]ncc5c4F)c2=O)[C@H](C)N(C(=O)c2cc4cc([C@H]5CCOC(C)(C)C5)ccc4n2[C@@]2(c4noc(=O)[nH]4)C[C@@H]2C)CC3)cc(C)c1F`
2. `Cc1cc(-n2nc3c(c2-n2ccn(-c4ccc5[nH]ncc5c4F)c2=O)[C@H](C)N(C(=O)c2cc4cc([C@H]5CCOC(C)(C)C5)ccc4n2[C@@]2(c4noc(=O)[nH]4)C[C@@H]2C)CC3)cc(C)c1F` → `C[C@H]1C[C@]1(c1noc(=O)[nH]1)n1c(C(=O)Cl)cc2cc([C@H]3CCOC(C)(C)C3)ccc21` + `Cc1cc(-n2nc3c(c2-n2ccn(-c4ccc5[nH]ncc5c4F)c2=O)[C@H](C)NCC3)cc(C)c1F`
3. `Cc1cc(-n2nc3c(c2-n2ccn(-c4ccc5[nH]ncc5c4F)c2=O)[C@H](C)NCC3)cc(C)c1F` → `Cc1cc(-n2nc3c(c2-n2ccn(-c4ccc5[nH]ncc5c4F)c2=O)[C@H](C)N(C(=O)O)CC3)cc(C)c1F`
4. `Cc1cc(-n2nc3c(c2-n2ccn(-c4ccc5[nH]ncc5c4F)c2=O)[C@H](C)N(C(=O)O)CC3)cc(C)c1F` → `COC(=O)N1CCc2nn(-c3cc(C)c(F)c(C)c3)c(-n3ccn(-c4ccc5[nH]ncc5c4F)c3=O)c2[C@@H]1C`

**Building blocks proposed:**
- ✓ `CI`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Acalabrutinib  (PubChem CID: 71226662)
**SMILES:** `CC#CC(=O)N1CCC[C@H]1C2=NC(=C3N2C=CN=C3N)C4=CC=C(C=C4)C(=O)NC5=CC=CC=N5`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC#CC(=O)N1CCC[C@H]1C2=NC(=C3N2C=CN=C3N)C4=CC=C(C=C4)C(=O)NC5=CC=CC=N5` → `CC#CC(=O)N1CCC[C@H]1c1nc(-c2ccc(C(N)=O)cc2)c2c(N)nccn12` + `Clc1ccccn1`
2. `CC#CC(=O)N1CCC[C@H]1c1nc(-c2ccc(C(N)=O)cc2)c2c(N)nccn12` → `CC#CC(=O)O` + `NC(=O)c1ccc(-c2nc([C@@H]3CCCN3)n3ccnc(N)c23)cc1`
3. `NC(=O)c1ccc(-c2nc([C@@H]3CCCN3)n3ccnc(N)c23)cc1` → `NC(=O)c1ccc(-c2nc([C@@H]3CCCN3)n3ccnc([N+](=O)[O-])c23)cc1`
4. `NC(=O)c1ccc(-c2nc([C@@H]3CCCN3)n3ccnc([N+](=O)[O-])c23)cc1` → `N` + `O=C(O)c1ccc(-c2nc([C@@H]3CCCN3)n3ccnc([N+](=O)[O-])c23)cc1`
5. `O=C(O)c1ccc(-c2nc([C@@H]3CCCN3)n3ccnc([N+](=O)[O-])c23)cc1` → `CCOC(=O)c1ccc(-c2nc([C@@H]3CCCN3)n3ccnc([N+](=O)[O-])c23)cc1`
6. `CCOC(=O)c1ccc(-c2nc([C@@H]3CCCN3)n3ccnc([N+](=O)[O-])c23)cc1` → `CCOC(=O)c1ccc(B(O)O)cc1` + `O=[N+]([O-])c1nccn2c([C@@H]3CCCN3)nc(Cl)c12`

**Building blocks proposed:**
- ✓ `Clc1ccccn1`
- ✓ `CC#CC(=O)O`
- ✓ `N`
- ✓ `CCOC(=O)c1ccc(B(O)O)cc1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Ibrutinib  (PubChem CID: 24821094)
**SMILES:** `C=CC(=O)N1CCC[C@H]1C2=NC3=C(N2)C=C(N=C3N)C4=CC=C(C=C4)OC5=CC=CC=C5`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C=CC(=O)N1CCC[C@H]1C2=NC3=C(N2)C=C(N=C3N)C4=CC=C(C=C4)OC5=CC=CC=C5` → `C=CC(=O)N1CCC[C@H]1c1nc2c(N)nc(-c3ccc(F)cc3)cc2[nH]1` + `Oc1ccccc1`
2. `C=CC(=O)N1CCC[C@H]1c1nc2c(N)nc(-c3ccc(F)cc3)cc2[nH]1` → `C=CC(=O)N1CCC[C@H]1c1nc2c(N)nc(Cl)cc2[nH]1` + `OB(O)c1ccc(F)cc1`
3. `C=CC(=O)N1CCC[C@H]1c1nc2c(N)nc(Cl)cc2[nH]1` → `C=CC(=O)N1CCC[C@H]1c1nc2c([N+](=O)[O-])nc(Cl)cc2[nH]1`
4. `C=CC(=O)N1CCC[C@H]1c1nc2c([N+](=O)[O-])nc(Cl)cc2[nH]1` → `C=CC(=O)Cl` + `O=[N+]([O-])c1nc(Cl)cc2[nH]c([C@@H]3CCCN3)nc12`
5. `O=[N+]([O-])c1nc(Cl)cc2[nH]c([C@@H]3CCCN3)nc12` → `O=P(Cl)(Cl)Cl` + `O=c1cc2[nH]c([C@@H]3CCCN3)nc2c([N+](=O)[O-])[nH]1`
6. `O=c1cc2[nH]c([C@@H]3CCCN3)nc2c([N+](=O)[O-])[nH]1` → `O=C(OCc1ccccc1)N1CCC[C@H]1c1nc2c([N+](=O)[O-])[nH]c(=O)cc2[nH]1`

**Building blocks proposed:**
- ✓ `Oc1ccccc1`
- ✓ `OB(O)c1ccc(F)cc1`
- ✓ `C=CC(=O)Cl`
- ✓ `O=P(Cl)(Cl)Cl`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Omeprazole  (PubChem CID: 4594)
**SMILES:** `COC1=CC2=C(NC3=CC=CC=C3N2CS(=O)C4=NC(=C(C=C4C)C)OC)C=C1OC`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `COC1=CC2=C(NC3=CC=CC=C3N2CS(=O)C4=NC(=C(C=C4C)C)OC)C=C1OC` → `CO` + `COc1cc2c(cc1O)Nc1ccccc1N2CS(=O)c1nc(OC)c(C)cc1C`
2. `COc1cc2c(cc1O)Nc1ccccc1N2CS(=O)c1nc(OC)c(C)cc1C` → `CO` + `COc1cc2c(cc1O)Nc1ccccc1N2CS(=O)c1nc(Cl)c(C)cc1C`
3. `COc1cc2c(cc1O)Nc1ccccc1N2CS(=O)c1nc(Cl)c(C)cc1C` → `COc1cc2c(cc1OCc1ccccc1)Nc1ccccc1N2CS(=O)c1nc(Cl)c(C)cc1C`
4. `COc1cc2c(cc1OCc1ccccc1)Nc1ccccc1N2CS(=O)c1nc(Cl)c(C)cc1C` → `COc1cc2c(cc1OCc1ccccc1)Nc1ccccc1N2CS(=O)c1[nH]c(=O)c(C)cc1C` + `O=P(Cl)(Cl)Cl`
5. `COc1cc2c(cc1OCc1ccccc1)Nc1ccccc1N2CS(=O)c1[nH]c(=O)c(C)cc1C` → `BrCc1ccccc1` + `COc1cc2c(cc1O)Nc1ccccc1N2CS(=O)c1[nH]c(=O)c(C)cc1C`
6. `COc1cc2c(cc1O)Nc1ccccc1N2CS(=O)c1[nH]c(=O)c(C)cc1C` → `COc1cc2c(cc1OC)N(CS(=O)c1[nH]c(=O)c(C)cc1C)c1ccccc1N2`

**Building blocks proposed:**
- ✓ `CO`
- ✓ `CO`
- ✓ `O=P(Cl)(Cl)Cl`
- ✓ `BrCc1ccccc1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Imatinib  (PubChem CID: 5291)
**SMILES:** `CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)CNC3=NC=NC(=C3)C4=CN=CC=C4`  
**Best route:** depth=4  solved=True  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)CNC3=NC=NC(=C3)C4=CN=CC=C4` → `Cc1ccc(C(=O)Cl)cc1` + `Nc1ccc(CNc2cc(-c3cccnc3)ncn2)cc1`
2. `Nc1ccc(CNc2cc(-c3cccnc3)ncn2)cc1` → `Nc1ccc(CNc2cc(Br)ncn2)cc1` + `OB(O)c1cccnc1`
3. `Nc1ccc(CNc2cc(Br)ncn2)cc1` → `Clc1cc(Br)ncn1` + `NCc1ccc(N)cc1`
4. `Clc1cc(Br)ncn1` → `Clc1ccncn1` + `O=C1CCC(=O)N1Br`

**Building blocks proposed:**
- ✓ `Cc1ccc(C(=O)Cl)cc1`
- ✓ `OB(O)c1cccnc1`
- ✓ `NCc1ccc(N)cc1`
- ✓ `Clc1ccncn1`
- ✓ `O=C1CCC(=O)N1Br`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Ozempic  (PubChem CID: 56843331)
**SMILES:** `C[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H](CC2=CN=CN2)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](CC3=CC=CC=C3)C(=O)N[C@@H](CC4=CC=CC=C4)C(=O)N[C@@H](CC5=CC=CC=C5)C(=O)N[C@@H](CC6=CC=CC=C6)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC7=CC=CC=C7)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC8=CC=CC=C8)C(=O)N[C@@H](CC9=CC=CC=C9)C(=O)N[C@@H](CC(C)C)C(=O)O`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H](CC2=CN=CN2)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](CC3=CC=CC=C3)C(=O)N[C@@H](CC4=CC=CC=C4)C(=O)N[C@@H](CC5=CC=CC=C5)C(=O)N[C@@H](CC6=CC=CC=C6)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CC7=CC=CC=C7)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC8=CC=CC=C8)C(=O)N[C@@H](CC9=CC=CC=C9)C(=O)N[C@@H](CC(C)C)C(=O)O` → `COC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCCN)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCN)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H](C)Cc1ccccc1`
2. `COC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCCN)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCN)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H](C)Cc1ccccc1` → `COC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](N)Cc1ccccc1` + `C[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)O`
3. `C[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)O` → `C[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCCNC(=O)OC(C)(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)O`
4. `C[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCCNC(=O)OC(C)(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)O` → `COC(=O)[C@H](CCCCN)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCNC(=O)OC(C)(C)C)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H](C)Cc1ccccc1`
5. `COC(=O)[C@H](CCCCN)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCNC(=O)OC(C)(C)C)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H](C)Cc1ccccc1` → `COC(=O)[C@H](CCCCN)NC(=O)[C@@H](N)Cc1ccccc1` + `C[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCCNC(=O)OC(C)(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O`
6. `C[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCCNC(=O)OC(C)(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O` → `CCOC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCCNC(=O)OC(C)(C)C)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H](C)Cc1ccccc1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Apixaban  (PubChem CID: 10182969)
**SMILES:** `COc1ccc(cc1)n2nc(C(=O)N)c3CCN(C(=O)c23)c4ccc(cc4)N5CCCCC5=O`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `COc1ccc(cc1)n2nc(C(=O)N)c3CCN(C(=O)c23)c4ccc(cc4)N5CCCCC5=O` → `COc1ccc(-n2nc(C(=O)O)c3c2C(=O)N(c2ccc(N4CCCCC4=O)cc2)CC3)cc1` + `N`
2. `COc1ccc(-n2nc(C(=O)O)c3c2C(=O)N(c2ccc(N4CCCCC4=O)cc2)CC3)cc1` → `COC(=O)c1nn(-c2ccc(OC)cc2)c2c1CCN(c1ccc(N3CCCCC3=O)cc1)C2=O`
3. `COC(=O)c1nn(-c2ccc(OC)cc2)c2c1CCN(c1ccc(N3CCCCC3=O)cc1)C2=O` → `COC(=O)c1nn(-c2ccc(OC)cc2)c2c1CCN(c1ccc(Br)cc1)C2=O` + `O=C1CCCCN1`
4. `COC(=O)c1nn(-c2ccc(OC)cc2)c2c1CCN(c1ccc(Br)cc1)C2=O` → `COC(=O)c1nn(-c2ccc(OC)cc2)c2c1CCN(c1ccccc1)C2=O` + `O=C1CCC(=O)N1Br`
5. `COC(=O)c1nn(-c2ccc(OC)cc2)c2c1CCN(c1ccccc1)C2=O` → `CO` + `COc1ccc(-n2nc(C(=O)O)c3c2C(=O)N(c2ccccc2)CC3)cc1`
6. `COc1ccc(-n2nc(C(=O)O)c3c2C(=O)N(c2ccccc2)CC3)cc1` → `CCOC(=O)c1nn(-c2ccc(OC)cc2)c2c1CCN(c1ccccc1)C2=O`

**Building blocks proposed:**
- ✓ `N`
- ✓ `O=C1CCCCN1`
- ✓ `O=C1CCC(=O)N1Br`
- ✓ `CO`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Palbocyclib  (PubChem CID: 5330286)
**SMILES:** `CC(=O)C1=NC2=C(N=C(N=C2N3CCNCC3)C4=NC=CC=C4)N(C1=O)C5CCCC5`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC(=O)C1=NC2=C(N=C(N=C2N3CCNCC3)C4=NC=CC=C4)N(C1=O)C5CCCC5` → `CC(=O)c1nc2c(N3CCN(C(=O)O)CC3)nc(-c3ccccn3)nc2n(C2CCCC2)c1=O`
2. `CC(=O)c1nc2c(N3CCN(C(=O)O)CC3)nc(-c3ccccn3)nc2n(C2CCCC2)c1=O` → `COC(=O)N1CCN(c2nc(-c3ccccn3)nc3c2nc(C(C)=O)c(=O)n3C2CCCC2)CC1`
3. `COC(=O)N1CCN(c2nc(-c3ccccn3)nc3c2nc(C(C)=O)c(=O)n3C2CCCC2)CC1` → `COC(=O)NCCN(CCBr)c1nc(-c2ccccn2)nc2c1nc(C(C)=O)c(=O)n2C1CCCC1`
4. `COC(=O)NCCN(CCBr)c1nc(-c2ccccn2)nc2c1nc(C(C)=O)c(=O)n2C1CCCC1` → `CC(=O)c1nc2c(N(CCBr)CCNC(=O)O)nc(-c3ccccn3)nc2n(C2CCCC2)c1=O` + `CI`
5. `CC(=O)c1nc2c(N(CCBr)CCNC(=O)O)nc(-c3ccccn3)nc2n(C2CCCC2)c1=O` → `CCOC(=O)NCCN(CCBr)c1nc(-c2ccccn2)nc2c1nc(C(C)=O)c(=O)n2C1CCCC1`
6. `CCOC(=O)NCCN(CCBr)c1nc(-c2ccccn2)nc2c1nc(C(C)=O)c(=O)n2C1CCCC1` → `CC(=O)c1nc2c(N(CCN)CCBr)nc(-c3ccccn3)nc2n(C2CCCC2)c1=O` + `CCOC(=O)Cl`

**Building blocks proposed:**
- ✓ `CI`
- ✓ `CCOC(=O)Cl`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Similar to Rivaroxaban  (PubChem CID: 9875401)
**SMILES:** `C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl` → `O=C(NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1)c1ccc(Cl)s1` + `O=C1COCCN1`
2. `O=C(NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1)c1ccc(Cl)s1` → `NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1` + `O=C(O)c1ccc(Cl)s1`
3. `NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1` → `[N-]=[N+]=NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1`
4. `[N-]=[N+]=NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1` → `Fc1ccc(Br)cc1` + `[N-]=[N+]=NC[C@H]1CNC(=O)O1`
5. `[N-]=[N+]=NC[C@H]1CNC(=O)O1` → `[N-]=[N+]=NC(=O)[C@H]1CNC(=O)O1`
6. `[N-]=[N+]=NC(=O)[C@H]1CNC(=O)O1` → `[N-]=[N+]=NC(=O)[C@H]1CN(C(=O)OCc2ccccc2)C(=O)O1`

**Building blocks proposed:**
- ✓ `O=C1COCCN1`
- ✓ `O=C(O)c1ccc(Cl)s1`
- ✓ `Fc1ccc(Br)cc1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Rivaroxaban  (PubChem CID: 9875401)
**SMILES:** `C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl` → `O=C(NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1)c1ccc(Cl)s1` + `O=C1COCCN1`
2. `O=C(NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1)c1ccc(Cl)s1` → `NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1` + `O=C(O)c1ccc(Cl)s1`
3. `NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1` → `[N-]=[N+]=NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1`
4. `[N-]=[N+]=NC[C@H]1CN(c2ccc(Br)cc2)C(=O)O1` → `Fc1ccc(Br)cc1` + `[N-]=[N+]=NC[C@H]1CNC(=O)O1`
5. `[N-]=[N+]=NC[C@H]1CNC(=O)O1` → `[N-]=[N+]=NC(=O)[C@H]1CNC(=O)O1`
6. `[N-]=[N+]=NC(=O)[C@H]1CNC(=O)O1` → `[N-]=[N+]=NC(=O)[C@H]1CN(C(=O)OCc2ccccc2)C(=O)O1`

**Building blocks proposed:**
- ✓ `O=C1COCCN1`
- ✓ `O=C(O)c1ccc(Cl)s1`
- ✓ `Fc1ccc(Br)cc1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Rivaroxaban  (PubChem CID: 9870771)
**SMILES:** `CCOC(=O)N1CCC[C@H]1C2=NC(=O)N(C3=CC=C(C=C3)N4CCOCC4=O)C5=CC(Cl)=CS5`  
*Skipped: invalid SMILES*

### Etoposide  (PubChem CID: 36462)
**SMILES:** `COC1=CC2=C(C=C1OC)C3=C(C(=O)OC4C(C(C(C(O4)CO)O)O)O)OC5=C3C=CC(=C5)O2`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `COC1=CC2=C(C=C1OC)C3=C(C(=O)OC4C(C(C(C(O4)CO)O)O)O)OC5=C3C=CC(=C5)O2` → `COCC1OC(OC(=O)c2oc3cc4ccc3c2-c2cc(OC)c(OC)cc2O4)C(O)C(O)C1O`
2. `COCC1OC(OC(=O)c2oc3cc4ccc3c2-c2cc(OC)c(OC)cc2O4)C(O)C(O)C1O` → `COCC1OC(OC(=O)c2oc3cc4ccc3c2-c2cc(OC)c(OC)cc2O4)C(OC)C(O)C1O`
3. `COCC1OC(OC(=O)c2oc3cc4ccc3c2-c2cc(OC)c(OC)cc2O4)C(OC)C(O)C1O` → `COCC1OC(OC(=O)c2oc3cc(Oc4cc(OC)c(OC)cc4B(O)O)ccc3c2Br)C(OC)C(O)C1O`
4. `COCC1OC(OC(=O)c2oc3cc(Oc4cc(OC)c(OC)cc4B(O)O)ccc3c2Br)C(OC)C(O)C1O` → `COCC1OC(OC(=O)c2cc3ccc(Oc4cc(OC)c(OC)cc4B(O)O)cc3o2)C(OC)C(O)C1O` + `O=C1CCC(=O)N1Br`
5. `COCC1OC(OC(=O)c2cc3ccc(Oc4cc(OC)c(OC)cc4B(O)O)cc3o2)C(OC)C(O)C1O` → `COCC1OC(OC(=O)c2cc3ccc(Oc4cc(OC)c(OC)cc4B(O)O)cc3o2)C(OC)C(O)C1OC`
6. `COCC1OC(OC(=O)c2cc3ccc(Oc4cc(OC)c(OC)cc4B(O)O)cc3o2)C(OC)C(O)C1OC` → `COCC1OC(OC(=O)c2cc3ccc(O)cc3o2)C(OC)C(O)C1OC` + `COc1cc(F)c(B(O)O)cc1OC`

**Building blocks proposed:**
- ✓ `O=C1CCC(=O)N1Br`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Similar to Mintedanib  (PubChem CID: 135423438)
**SMILES:** `CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O` → `CNC(=O)CN1CCN(C)CC1` + `COC(=O)c1ccc2c(C(=Nc3ccc(Br)cc3)c3ccccc3)c(O)[nH]c2c1`
2. `COC(=O)c1ccc2c(C(=Nc3ccc(Br)cc3)c3ccccc3)c(O)[nH]c2c1` → `COC(=O)c1ccc2c(C(=Nc3ccc(Br)cc3)c3ccccc3)c(OC)[nH]c2c1`
3. `COC(=O)c1ccc2c(C(=Nc3ccc(Br)cc3)c3ccccc3)c(OC)[nH]c2c1` → `COC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(OC)[nH]c2c1` + `O=C1CCC(=O)N1Br`
4. `COC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(OC)[nH]c2c1` → `CO` + `COc1[nH]c2cc(C(=O)O)ccc2c1C(=Nc1ccccc1)c1ccccc1`
5. `COc1[nH]c2cc(C(=O)O)ccc2c1C(=Nc1ccccc1)c1ccccc1` → `CCOC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(OC)[nH]c2c1`
6. `CCOC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(OC)[nH]c2c1` → `CCOC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(Cl)[nH]c2c1` + `CO`

**Building blocks proposed:**
- ✓ `O=C1CCC(=O)N1Br`
- ✓ `CO`
- ✓ `CO`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Mintedanib  (PubChem CID: 135423438)
**SMILES:** `CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O` → `CNC(=O)CN1CCN(C)CC1` + `COC(=O)c1ccc2c(C(=Nc3ccc(Br)cc3)c3ccccc3)c(O)[nH]c2c1`
2. `COC(=O)c1ccc2c(C(=Nc3ccc(Br)cc3)c3ccccc3)c(O)[nH]c2c1` → `COC(=O)c1ccc2c(C(=Nc3ccc(Br)cc3)c3ccccc3)c(OC)[nH]c2c1`
3. `COC(=O)c1ccc2c(C(=Nc3ccc(Br)cc3)c3ccccc3)c(OC)[nH]c2c1` → `COC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(OC)[nH]c2c1` + `O=C1CCC(=O)N1Br`
4. `COC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(OC)[nH]c2c1` → `CO` + `COc1[nH]c2cc(C(=O)O)ccc2c1C(=Nc1ccccc1)c1ccccc1`
5. `COc1[nH]c2cc(C(=O)O)ccc2c1C(=Nc1ccccc1)c1ccccc1` → `CCOC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(OC)[nH]c2c1`
6. `CCOC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(OC)[nH]c2c1` → `CCOC(=O)c1ccc2c(C(=Nc3ccccc3)c3ccccc3)c(Cl)[nH]c2c1` + `CO`

**Building blocks proposed:**
- ✓ `O=C1CCC(=O)N1Br`
- ✓ `CO`
- ✓ `CO`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Methoxybiphenyl_Sulfonamide_Amidoxime  (PubChem CID: 9870771)
**SMILES:** `COC1=CC=C(C=C1)C2=CC=C(C=C2)S(=O)(=O)N[C@H]3CCN(C3=O)CC(=O)NCCON=C(N)N`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `COC1=CC=C(C=C1)C2=CC=C(C=C2)S(=O)(=O)N[C@H]3CCN(C3=O)CC(=O)NCCON=C(N)N` → `COc1ccc(B(O)O)cc1` + `NC(N)=NOCCNC(=O)CN1CC[C@H](NS(=O)(=O)c2ccc(Br)cc2)C1=O`
2. `NC(N)=NOCCNC(=O)CN1CC[C@H](NS(=O)(=O)c2ccc(Br)cc2)C1=O` → `NC(N)=NOCCNC(=O)CNCC[C@H](NS(=O)(=O)c1ccc(Br)cc1)C(=O)Cl`
3. `NC(N)=NOCCNC(=O)CNCC[C@H](NS(=O)(=O)c1ccc(Br)cc1)C(=O)Cl` → `NCCON=C(N)N` + `O=C(O)CNCC[C@H](NS(=O)(=O)c1ccc(Br)cc1)C(=O)Cl`
4. `O=C(O)CNCC[C@H](NS(=O)(=O)c1ccc(Br)cc1)C(=O)Cl` → `COC(=O)CNCC[C@H](NS(=O)(=O)c1ccc(Br)cc1)C(=O)Cl`
5. `COC(=O)CNCC[C@H](NS(=O)(=O)c1ccc(Br)cc1)C(=O)Cl` → `COC(=O)CNCC[C@H](N)C(=O)Cl` + `O=S(=O)(Cl)c1ccc(Br)cc1`
6. `COC(=O)CNCC[C@H](N)C(=O)Cl` → `CO` + `N[C@@H](CCNCC(=O)O)C(=O)Cl`

**Building blocks proposed:**
- ✓ `COc1ccc(B(O)O)cc1`
- ✓ `O=S(=O)(Cl)c1ccc(Br)cc1`
- ✓ `CO`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Similar to Losartan  (PubChem CID: 3961)
**SMILES:** `CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl` → `CCCCc1nc(Cl)c(C(=O)O)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1`
2. `CCCCc1nc(Cl)c(C(=O)O)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1` → `CCCCc1nc(Cl)c(C(=O)OC)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1`
3. `CCCCc1nc(Cl)c(C(=O)OC)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1` → `CCCCc1nc(Cl)c(C(=O)OC)n1Cc1ccc(Br)cc1` + `OB(O)c1ccccc1-c1nn[nH]n1`
4. `CCCCc1nc(Cl)c(C(=O)OC)n1Cc1ccc(Br)cc1` → `CCCCc1nc(Cl)c(C(=O)O)n1Cc1ccc(Br)cc1` + `CO`
5. `CCCCc1nc(Cl)c(C(=O)O)n1Cc1ccc(Br)cc1` → `CCCCc1nc(Cl)c(C(=O)OCC)n1Cc1ccc(Br)cc1`
6. `CCCCc1nc(Cl)c(C(=O)OCC)n1Cc1ccc(Br)cc1` → `BrCc1ccc(Br)cc1` + `CCCCc1nc(Cl)c(C(=O)OCC)[nH]1`

**Building blocks proposed:**
- ✓ `OB(O)c1ccccc1-c1nn[nH]n1`
- ✓ `CO`
- ✓ `BrCc1ccc(Br)cc1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Losartan  (PubChem CID: 3961)
**SMILES:** `Cc1ccc(cc1)C(=O)N2C(=NNC2c3ccccc3Cn4cc(c(n4)C)Cl)C`  
**Best route:** depth=5  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `Cc1ccc(cc1)C(=O)N2C(=NNC2c3ccccc3Cn4cc(c(n4)C)Cl)C` → `CC1=NNC(c2ccccc2Cn2cc(Cl)c(C)n2)N1` + `Cc1ccc(C(=O)Cl)cc1`
2. `CC1=NNC(c2ccccc2Cn2cc(Cl)c(C)n2)N1` → `CC1=NNC(c2ccccc2CCl)N1` + `Cc1n[nH]cc1Cl`
3. `CC1=NNC(c2ccccc2CCl)N1` → `CC1=NNC(c2ccccc2CO)N1` + `ClCCl`
4. `CC1=NNC(c2ccccc2CO)N1` → `CC1=NNC(c2ccccc2C(=O)O)N1`
5. `CC1=NNC(c2ccccc2C(=O)O)N1` → `COC(=O)c1ccccc1C1NN=C(C)N1`

**Building blocks proposed:**
- ✓ `Cc1ccc(C(=O)Cl)cc1`
- ✓ `Cc1n[nH]cc1Cl`
- ✓ `ClCCl`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Venetoclax  (PubChem CID: 49846579)
**SMILES:** `CC1(CCC(=C(C1)C2=CC=C(C=C2)Cl)CN3CCN(CC3)C4=CC(=C(C=C4)C(=O)NS(=O)(=O)C5=CC(=C(C=C5)NCC6CCOCC6)[N+](=O)[O-])OC7=CN=C8C(=C7)C=CN8)C`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `CC1(CCC(=C(C1)C2=CC=C(C=C2)Cl)CN3CCN(CC3)C4=CC(=C(C=C4)C(=O)NS(=O)(=O)C5=CC(=C(C=C5)NCC6CCOCC6)[N+](=O)[O-])OC7=CN=C8C(=C7)C=CN8)C` → `CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)c(F)c3)CC2)=C(c2ccc(Cl)cc2)C1` + `Oc1cnc2[nH]ccc2c1`
2. `CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)c(F)c3)CC2)=C(c2ccc(Cl)cc2)C1` → `CC1(C)CCC(CN2CCN(c3ccc(C(N)=O)c(F)c3)CC2)=C(c2ccc(Cl)cc2)C1` + `O=[N+]([O-])c1cc(S(=O)(=O)Cl)ccc1NCC1CCOCC1`
3. `O=[N+]([O-])c1cc(S(=O)(=O)Cl)ccc1NCC1CCOCC1` → `O=[N+]([O-])c1cc(S(=O)(=O)Cl)ccc1NCC(CCO)CCBr`
4. `O=[N+]([O-])c1cc(S(=O)(=O)Cl)ccc1NCC(CCO)CCBr` → `NCC(CCO)CCBr` + `O=[N+]([O-])c1cc(S(=O)(=O)Cl)ccc1Cl`
5. `NCC(CCO)CCBr` → `COCCC(CN)CCBr`
6. `COCCC(CN)CCBr` → `COCCC(CCBr)CNC(=O)OC(C)(C)C`

**Building blocks proposed:**
- ✓ `Oc1cnc2[nH]ccc2c1`
- ✓ `O=[N+]([O-])c1cc(S(=O)(=O)Cl)ccc1Cl`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Similar to Cefepime  (PubChem CID: 5479537)
**SMILES:** `C[N+]1(CCCC1)CC2=C(N3[C@@H]([C@@H](C3=O)NC(=O)/C(=N\OC)/C4=CSC(=N4)N)SC2)C(=O)[O-]`  
**Best route:** depth=5  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C[N+]1(CCCC1)CC2=C(N3[C@@H]([C@@H](C3=O)NC(=O)/C(=N\OC)/C4=CSC(=N4)N)SC2)C(=O)[O-]` → `CCOC(=O)/C(=N\OC)c1csc(N)n1` + `C[N+]1(CC2=C(C(=O)[O-])N3C(=O)[C@@H](N)[C@H]3SC2)CCCC1`
2. `C[N+]1(CC2=C(C(=O)[O-])N3C(=O)[C@@H](N)[C@H]3SC2)CCCC1` → `C[N+]1(CC2=C(C(=O)[O-])N[C@@H]([C@H](N)C(=O)Cl)SC2)CCCC1`
3. `C[N+]1(CC2=C(C(=O)[O-])N[C@@H]([C@H](N)C(=O)Cl)SC2)CCCC1` → `C[N+]1(CC2=C(C(=O)[O-])N[C@@H]([C@@H](C(=O)Cl)N3C(=O)c4ccccc4C3=O)SC2)CCCC1`
4. `C[N+]1(CC2=C(C(=O)[O-])N[C@@H]([C@@H](C(=O)Cl)N3C(=O)c4ccccc4C3=O)SC2)CCCC1` → `C[N+]1(CC2=C(C(=O)[O-])N[C@@H]([C@H](NC(=O)c3ccccc3C(=O)O)C(=O)Cl)SC2)CCCC1`
5. `C[N+]1(CC2=C(C(=O)[O-])N[C@@H]([C@H](NC(=O)c3ccccc3C(=O)O)C(=O)Cl)SC2)CCCC1` → `CCOC(=O)c1ccccc1C(=O)N[C@H](C(=O)Cl)[C@@H]1NC(C(=O)[O-])=C(C[N+]2(C)CCCC2)CS1`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Cefepime  (PubChem CID: 49846579)
**SMILES:** `None`  
*Skipped: no SMILES*

### Similar to Lorazepam  (PubChem CID: 3958)
**SMILES:** `C1=CC=C(C(=C1)C2=NC(C(=O)NC3=C2C=C(C=C3)Cl)O)Cl`  
**Best route:** depth=5  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C1=CC=C(C(=C1)C2=NC(C(=O)NC3=C2C=C(C=C3)Cl)O)Cl` → `COC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O`
2. `COC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O` → `COC(N=C(c1cc(Cl)ccc1N)c1ccccc1Cl)C(=O)O`
3. `COC(N=C(c1cc(Cl)ccc1N)c1ccccc1Cl)C(=O)O` → `COC(N=C(c1ccccc1Cl)c1cc(Cl)ccc1[N+](=O)[O-])C(=O)O`
4. `COC(N=C(c1ccccc1Cl)c1cc(Cl)ccc1[N+](=O)[O-])C(=O)O` → `COC(N=C(c1cccc(Cl)c1)c1ccccc1Cl)C(=O)O` + `O=[N+]([O-])O`
5. `COC(N=C(c1cccc(Cl)c1)c1ccccc1Cl)C(=O)O` → `COC(=O)C(N=C(c1cccc(Cl)c1)c1ccccc1Cl)OC`

**Building blocks proposed:**
- ✓ `O=[N+]([O-])O`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

### Lorazepam  (PubChem CID: 3958)
**SMILES:** `None`  
*Skipped: no SMILES*

### Vepdegestrant  (PubChem CID: 134562533)
**SMILES:** `C1CC2=C(C=CC(=C2)O)[C@H]([C@H]1C3=CC=CC=C3)C4=CC=C(C=C4)N5CCC(CC5)CN6CCN(CC6)C7=CC8=C(C=C7)C(=O)N(C8)[C@H]9CCC(=O)NC9=O`  
**Best route:** depth=6  solved=False  score=0  ⚠ cyclic

**Reactions (retrosynthetic direction, target → reactants):**
1. `C1CC2=C(C=CC(=C2)O)[C@H]([C@H]1C3=CC=CC=C3)C4=CC=C(C=C4)N5CCC(CC5)CN6CCN(CC6)C7=CC8=C(C=C7)C(=O)N(C8)[C@H]9CCC(=O)NC9=O` → `O=C1CC[C@H](N2Cc3cc(N4CCN(CC5CCNCC5)CC4)ccc3C2=O)C(=O)N1` + `Oc1ccc2c(c1)CC[C@H](c1ccccc1)[C@@H]2c1ccc(Br)cc1`
2. `Oc1ccc2c(c1)CC[C@H](c1ccccc1)[C@@H]2c1ccc(Br)cc1` → `COc1ccc2c(c1)CC[C@H](c1ccccc1)[C@@H]2c1ccc(Br)cc1`
3. `COc1ccc2c(c1)CC[C@H](c1ccccc1)[C@@H]2c1ccc(Br)cc1` → `BrBr` + `COc1ccc2c(c1)CC[C@H](c1ccccc1)[C@@H]2c1ccccc1`
4. `COc1ccc2c(c1)CC[C@H](c1ccccc1)[C@@H]2c1ccccc1` → `CO` + `Oc1ccc2c(c1)CC[C@H](c1ccccc1)[C@@H]2c1ccccc1`
5. `Oc1ccc2c(c1)CC[C@H](c1ccccc1)[C@@H]2c1ccccc1` → `O=C1C[C@H](c2ccccc2)[C@H](c2ccccc2)c2ccc(O)cc21`
6. `O=C1C[C@H](c2ccccc2)[C@H](c2ccccc2)c2ccc(O)cc21` → `COc1ccc2c(c1)C(=O)C[C@H](c1ccccc1)[C@@H]2c1ccccc1`

**Building blocks proposed:**
- ✓ `BrBr`
- ✓ `CO`

> ⚠ **Cyclic route:** a molecule appears as both a target and a reactant — beam search looped without finding purchasable leaves.

---

## Notes on Model Behavior

- **score=0.0** reflects floating-point underflow of `trajectory_prob` (product of per-step probabilities across 6 steps), not zero probability. These routes are still chemically valid proposals.
- **Cyclic routes** occur when the model repeatedly applies ester hydrolysis ↔ esterification or similar reversible transforms, indicating the beam search depth limit (6) was reached without finding a purchasable route.
- **depth=0** means the target SMILES itself matches a known building block in the PaRoutes training set.

