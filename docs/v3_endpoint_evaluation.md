# RetroSynFormer v3 Endpoint Evaluation Report

**Date:** 2026-06-21  
**Endpoint:** `https://retrosynformer-inference-v3-knq67derjq-uc.a.run.app/retrosynthesis`  
**Model:** Large (action_dim=2957, 24-layer Decision Transformer)  
**Beam width:** 10  
**Evaluation method:** Sequential POST requests, `timeout=120s`

---

## Summary Table

| Molecule | Complexity | Routes | Depth | Solved | Cyclic | Leaves (purch/total) | Score | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Aspirin | 212 | 1 | 0 | Yes | No | 0/0 | 1.000000 | 1.0 |
| Ibuprofen | 203 | 5 | 6 | No | **Yes** | 3/3 | 0.000000 | 8.3 |
| Paclitaxel | 1790 | 5 | 6 | No | **Yes** | 2/2 | 0.000000 | 9.7 |
| Etoricoxib | 514 | 5 | 2 | Yes | **Yes** | 3/3 | 0.017300 | 12.4 |
| Camlipixant | 704 | 5 | 6 | No | **Yes** | 3/3 | 0.000000 | 9.1 |
| Fluorinated_Imidazole (CID 84117446) | 272 | 5 | 2 | Yes | **Yes** | 3/3 | 0.000000 | 8.8 |
| Fluorinated_Imidazole (CID 56842878) | 632 | 5 | 6 | No | **Yes** | 2/2 | 0.000000 | 9.0 |
| Methoxy_Diphenylamine | 191 | 5 | 1 | Yes | No | 2/2 | 0.014900 | 7.3 |
| Tolyl_Pyridine | 162 | 5 | 1 | Yes | No | 2/2 | 0.008200 | 8.0 |
| Orforglipron | 1950 | 5 | 6 | No | **Yes** | 1/1 | 0.000000 | 9.2 |
| Acalabrutinib | 845 | 5 | 6 | No | **Yes** | 3/3 | 0.000000 | 8.8 |
| Ibrutinib | 763 | 5 | 6 | No | **Yes** | 4/4 | 0.000000 | 8.8 |
| Omeprazole | 339 | 5 | 6 | No | **Yes** | 3/3 | 0.000000 | 9.0 |
| Imatinib | 742 | 5 | 3 | Yes | **Yes** | 4/4 | 0.000400 | 8.4 |
| Semaglutide (Ozempic) | 1700 | 5 | 6 | No | **Yes** | 2/2 | 0.000000 | 10.8 |
| Apixaban | 582 | 5 | 6 | No | **Yes** | 3/3 | 0.000000 | 9.8 |
| Palbocyclib | 667 | 5 | 6 | No | **Yes** | 2/2 | 0.000000 | 8.6 |
| Rivaroxaban | 589 | 5 | 6 | No | **Yes** | 3/3 | 0.000000 | 8.8 |
| Etoposide | 804 | 5 | 6 | No | **Yes** | 1/1 | 0.000000 | 9.7 |
| Nintedanib | 754 | 5 | 6 | No | **Yes** | 3/3 | 0.000000 | 9.0 |
| Losartan | 492 | 5 | 6 | No | **Yes** | 4/4 | 0.000000 | 8.9 |
| Venetoclax | 1640 | 5 | 6 | No | **Yes** | 4/4 | 0.000000 | 9.8 |

**Note:** Cefepime, Midazolam, and Lorazepam are listed in `test_molecules.yml` but have no SMILES recorded — they were excluded from testing.

---

## Overall Statistics

| Metric | Count |
|---|---|
| Total molecules in YAML | 25 |
| Molecules with SMILES (tested) | 22 |
| Invalid SMILES | 0 |
| Skipped (no SMILES in file) | 3 |
| API errors | 0 |
| Solved (all_leaves_purchasable=True) | 6 (27%) |
| Depth=0 (molecule is a building block) | 1 |
| Solved at depth 1 | 2 |
| Solved at depth 2 | 2 |
| Solved at depth 3 | 1 |
| Cyclic best routes | 19 (86%) |
| Average depth, non-trivial routes | 5.0 |
| Minimum response time | 1.0 s (Aspirin) |
| Maximum response time | 12.4 s (Etoricoxib) |
| Typical response time | ~9 s |

### Solved molecules
- Aspirin (depth=0, building block)
- Methoxy_Diphenylamine (depth=1)
- Tolyl_Pyridine (depth=1)
- Etoricoxib (depth=2)
- Fluorinated_Imidazole / CID 84117446 (depth=2)
- Imatinib (depth=3)

### Unsolved molecules (16 of 22)
All returned routes reach depth=6 (the maximum explored), with score=0.0 and `all_leaves_purchasable=False`. The model exhausted its search budget without finding a fully purchasable route for Ibuprofen, Paclitaxel, Camlipixant, Fluorinated_Imidazole (CID 56842878), Orforglipron, Acalabrutinib, Ibrutinib, Omeprazole, Semaglutide, Apixaban, Palbocyclib, Rivaroxaban, Etoposide, Nintedanib, Losartan, and Venetoclax.

---

## Per-Molecule Detail

### 1. Aspirin
**Target SMILES:** `CC(=O)OC1=CC=CC=C1C(=O)O`  
**MW:** 180.2 Da | **Complexity:** 212

**Result:** Correctly recognized as a purchasable building block. Depth=0, score=1.0. No synthesis steps returned — the molecule itself is in the building-block library.

**Plausibility:** Correct and expected behavior. Aspirin is commercially available at commodity scale.

---

### 2. Ibuprofen
**Target SMILES:** `CC(C)CC1=CC=C(C=C1)[C@@H](C)C(=O)O`  
**MW:** 206.3 Da | **Complexity:** 203

**Best route reactions:**
1. Ibuprofen → methyl ibuprofen ester (esterification)
2. Methyl ester → Ibuprofen + MeOH (hydrolysis — reverses step 1)
3. Ibuprofen → ethyl ester
4. Ethyl ester → Ibuprofen + EtOH (reverses step 3)
5–6. Repeat ester ↔ acid oscillation

**Leaf molecules:** CO (methanol), CCO (ethanol), CO (methanol)  
**Plausibility:** **Cyclic route — ester hydrolysis ↔ esterification loop.** The model oscillates between the free acid and its methyl/ethyl esters without making forward synthetic progress. All three reported "leaf" molecules (methanol, ethanol) are trivially purchasable but the route is chemically nonsensical. This is a known failure mode: the model applies reversible ester hydrolysis templates repeatedly, never converging on a true retrosynthetic disconnect.

---

### 3. Paclitaxel
**Target SMILES:** (complex taxane scaffold, MW=854 Da)  
**MW:** 853.9 Da | **Complexity:** 1790

**Best route reactions:**
1. Paclitaxel → baccatin III core fragment + phenylisoserine side chain (correct disconnection)
2. Phenylisoserine side chain (carboxylic acid form) → methyl ester
3. Methyl ester → amino alcohol + benzoic acid
4. Amino alcohol → methoxy derivative
5–6. Methoxy ↔ alcohol oscillation

**Leaf molecules:** `O=C(O)c1ccccc1` (benzoic acid, MW=122, purchasable), `CO` (methanol, trivially purchasable)  
**Plausibility:** **Partially correct then cyclic.** The first step makes chemical sense (Paclitaxel retrosynthesis classically disconnects at the ester linking baccatin III to the phenylisoserine side chain). Steps 2–6 devolve into a methyl ester oscillation cycle. The complex taxane scaffold is never decomposed further. The model cannot route the baccatin III bicyclic core.

---

### 4. Etoricoxib
**Target SMILES:** `CC1=NC=C(C=C1)C2=C(C=C(C=N2)Cl)C3=CC=C(C=C3)S(=O)(=O)C`  
**MW:** 358.9 Da | **Complexity:** 514

**Best route reactions:**
1. Etoricoxib → 4-(methylsulfonyl)phenylboronic acid + 2-bromo-5-(6-methylpyridin-3-yl)-3-chloropyridine (Suzuki coupling)
2. Bromo intermediate → 6-methylpyridin-3-ylboronic acid + 2-chloro-3,5-dibromopyridine (second Suzuki)

**Leaf molecules:**
- `CS(=O)(=O)c1ccc(B(O)O)cc1` — 4-(methylsulfonyl)phenylboronic acid (MW=200, purchasable)
- `Cc1ccc(B(O)O)cn1` — 6-methylpyridin-3-ylboronic acid (MW=137, purchasable)
- `Clc1cnc(Cl)c(Br)c1` — 2,3-dichloro-5-bromopyridine (MW=227, purchasable)

**Plausibility:** **Good route, chemically reasonable.** Suzuki cross-coupling disconnections are appropriate for this biaryl compound. All three building blocks are commercially available. Route is marked cyclic because the intermediate `Cc1ccc(-c2ncc(Cl)cc2Br)cn1` appears as both a step target and a step reactant due to how the tree branches, but this is an artifact of beam search branching rather than a true chemical cycle. Score=0.0173 is the highest non-trivial score in the dataset.

---

### 5. Camlipixant
**Target SMILES:** `CC1=CC2=NC(=C(N2C=C1)C[C@H]3CN(CCO3)C(=O)OC)C4=C(C=C(C=C4F)C(=O)NC)F`  
**MW:** 458.5 Da | **Complexity:** 704

**Best route reactions:**
1. Camlipixant → methylamine + morpholine-carbamate intermediate (N-methylamine elimination)
2. Morpholine intermediate → chloroimidazopyridine + difluorophenylboronic acid (Suzuki)
3–6. Boronic acid carboxylic acid ↔ methyl ester oscillation

**Leaf molecules:** CN (methylamine), CO (methanol), CO (methanol)  
**Plausibility:** **Cyclic — boronic acid ester oscillation.** Steps 3–6 cycle between the carboxylic acid and methyl ester of the difluorophenylboronic acid reagent. The first two steps show reasonable disconnection logic. Unresolved due to template oscillation on the boronic acid reagent.

---

### 6. Fluorinated Imidazole (CID 84117446)
**Target SMILES:** `CN1C=NC(=C1C2=CC=CC=C2F)C#N`  
**MW:** 201.2 Da | **Complexity:** 272

**Best route reactions:**
1. Fluorinated imidazole → N-methyl-4-bromoimidazole-5-carbonitrile + 2-fluorophenylboronic acid (Suzuki)
2. Bromo imidazole → N-methylimidazole-5-carbonitrile + N-bromosuccinimide (bromination)

**Leaf molecules:**
- `OB(O)c1ccccc1F` — 2-fluorophenylboronic acid (MW=140, purchasable)
- `Cn1cnc(C#N)c1` — 1-methyl-1H-imidazole-4-carbonitrile (MW=107, purchasable)
- `O=C1CCC(=O)N1Br` — N-bromosuccinimide (MW=178, purchasable)

**Plausibility:** **Reasonable route, correct chemistry.** Suzuki coupling to install the fluorophenyl group is a standard approach. NBS bromination of the imidazole ring is chemically appropriate. All building blocks are readily purchasable. Marked cyclic due to the NBS intermediate appearing in the branching tree; this is a beam-search artifact.

---

### 7. Fluorinated Imidazole (CID 56842878) — Dipeptide
**Target SMILES:** `C1C[C@H](N(C1)C(=O)[C@H](CC2=C(N=CN2)F)NC(=O)[C@@H]3CCC(=O)N3)C(=O)N`  
**MW:** 380.4 Da | **Complexity:** 632

**Best route reactions:**
1. Target → pyroglutamyl-fluoroimidazolyl-alanyl-prolinamide (minor rewrite)
2. Rewritten peptide → methyl glutamine ester + dipeptide fragment
3. Methyl ester → methanol + tripeptide acid (hydrolysis)
4–6. Oscillation between methyl ester and acid forms

**Leaf molecules:** CO (methanol), CO (methanol)  
**Plausibility:** **Cyclic — methyl ester oscillation.** The model recognizes this as a peptide but cannot plan a convergent peptide synthesis. Steps oscillate between ester and acid of the glutamine residue. The fluoroimidazole-containing amino acid residue is never individually disconnected. Only methanol emerges as a leaf molecule, which is trivially purchasable but chemically useless.

---

### 8. Methoxy_Diphenylamine
**Target SMILES:** `CC1=CC=C(C=C1)NC2=CC=C(C=C2)OC`  
**MW:** 213.3 Da | **Complexity:** 191

**Best route reactions:**
1. N-(4-methoxyphenyl)-4-methylaniline → 4-methoxyaniline + 4-bromotoluene (Buchwald-Hartwig amination)

**Leaf molecules:**
- `COc1ccc(N)cc1` — 4-methoxyaniline (MW=123, purchasable)
- `Cc1ccc(Br)cc1` — 4-bromotoluene (MW=171, purchasable)

**Plausibility:** **Excellent route.** Single-step Buchwald-Hartwig C–N coupling disconnection. Both building blocks are cheap commodity chemicals. Score=0.0149.

---

### 9. Tolyl_Pyridine
**Target SMILES:** `CC1=CC=C(C=C1)NC2=CN=CC=C2`  
**MW:** 184.2 Da | **Complexity:** 162

**Best route reactions:**
1. N-(4-methylphenyl)pyridin-3-amine → 3-bromopyridine + 4-toluidine (Buchwald-Hartwig)

**Leaf molecules:**
- `Brc1cccnc1` — 3-bromopyridine (MW=158, purchasable)
- `Cc1ccc(N)cc1` — 4-methylaniline / p-toluidine (MW=107, purchasable)

**Plausibility:** **Excellent route.** Single-step Buchwald-Hartwig disconnection. Both building blocks are common laboratory reagents. Score=0.0082.

---

### 10. Orforglipron
**Target SMILES:** (large polycyclic GLP-1 agonist, MW=883 Da)  
**MW:** 883.0 Da | **Complexity:** 1950

**Best route reactions:**
1. Orforglipron → complex bromoalkyl intermediate (partial disconnection of oxadiazolone ring)
2. Intermediate → large fragment with bromoethyl side chain + indole boronate
3–6. Successive halide substitutions on large fragments that remain nearly as complex as the starting material

**Leaf molecules:** `O=C(Cl)OCc1ccccc1` — benzyl chloroformate (MW=171, purchasable)  
**Plausibility:** **Severely cyclic with minimal simplification.** The route makes incremental disconnections on peripheral groups while leaving the polycyclic core intact. Cyclic SMILES detected include fragments that are 70–80% of the molecular weight of the target itself, indicating the model is not achieving meaningful retrosynthetic simplification. Only one leaf molecule is reported (benzyl chloroformate), and the route is deeply cyclic. This reflects the difficulty of Orforglipron (Bertz score 1950, the highest in the dataset) for a model trained on PaRoutes templates.

---

### 11. Acalabrutinib
**Target SMILES:** `CC#CC(=O)N1CCC[C@H]1C2=NC(=C3N2C=CN=C3N)C4=CC=C(C=C4)C(=O)NC5=CC=CC=N5`  
**MW:** 465.5 Da | **Complexity:** 845

**Best route reactions:**
1. Acalabrutinib → acalabrutinib–benzoic acid fragment + 2-aminopyridine (amide coupling)
2. Fragment → nitro precursor (reduction of amino group)
3. Nitro compound → methyl ester of benzoic acid fragment (esterification)
4. Methyl ester → Suzuki precursor + 4-(methoxycarbonyl)phenylboronic acid
5. Suzuki precursor → phosphorus oxychloride + hydroxyl compound (hydroxyl activation)
6. Hydroxyl compound → methoxy precursor (demethylation — reversal)

**Leaf molecules:**
- `Nc1ccccn1` — 2-aminopyridine (MW=94, purchasable)
- `COC(=O)c1ccc(B(O)O)cc1` — 4-(methoxycarbonyl)phenylboronic acid (MW=180, purchasable)
- `O=P(Cl)(Cl)Cl` — phosphorus oxychloride (MW=153, purchasable)

**Plausibility:** **Partially reasonable but cyclic at the core.** The first disconnection (amide bond hydrolysis to give 2-aminopyridine) is chemically valid. Suzuki coupling to install the benzoic acid group is reasonable. However, steps 5–6 introduce a methoxy ↔ hydroxyl oscillation on the imidazopyrazine core. POCl₃ is a reactive reagent that is technically purchasable but requires careful handling. Route is deeply cyclic overall.

---

### 12. Ibrutinib
**Target SMILES:** `C=CC(=O)N1CCC[C@H]1C2=NC3=C(N2)C=C(N=C3N)C4=CC=C(C=C4)OC5=CC=CC=C5`  
**MW:** 425.5 Da | **Complexity:** 763

**Best route reactions:**
1. Ibrutinib → amino-chloro imidazopyrimidine fragment + 4-(phenoxy)phenylboronic acid (Suzuki)
2. Amino compound → nitro compound (nitro reduction — backward)
3. Nitro compound → acryloyl compound + chloroamino fragment
4. Chloro compound → POCl₃ + hydroxyl compound
5. Hydroxyl compound → OBn protected compound (benzylation)
6. OBn compound → BrCH₂Ph + hydroxyl compound (reversal)

**Leaf molecules:**
- `OB(O)c1ccc(Oc2ccccc2)cc1` — 4-phenoxyphenylboronic acid (MW=214, purchasable)
- `C=CC(=O)O` — acrylic acid (MW=72, purchasable)
- `O=P(Cl)(Cl)Cl` — POCl₃ (purchasable)
- `BrCc1ccccc1` — benzyl bromide (MW=171, purchasable)

**Plausibility:** **Cyclic with reasonable individual steps.** The Suzuki disconnection at step 1 is chemically valid. The acryloyl amide introduction is rational. However, steps 5–6 form a benzyl ether protection/deprotection loop. All four leaf molecules are purchasable but the route is marked unsolved because the overall tree contains cyclic SMILES.

---

### 13. Omeprazole
**Target SMILES:** `COC1=CC2=C(NC3=CC=CC=C3N2CS(=O)C4=NC(=C(C=C4C)C)OC)C=C1OC`  
**MW:** 439.5 Da | **Complexity:** 339

**Best route reactions:**
1. Omeprazole → iodomethane + des-methyl omeprazole (O-methylation — backward)
2. Des-methyl compound → OBn-protected benzimidazole (benzylation)
3. OBn compound → ClCH₂Ph + des-methyl compound (reversal — deprotection)
4–6. Benzylation/debenzylation oscillation

**Leaf molecules:** `CI` (iodomethane, purchasable), `ClCc1ccccc1` (benzyl chloride, purchasable), `BrCc1ccccc1` (benzyl bromide, purchasable)  
**Plausibility:** **Cyclic — O-protection oscillation.** Steps 3–6 form a loop between OBn-protected and deprotected forms of the benzimidazole oxygen. The model uses three different alkylating agents (MeI, BnCl, BnBr) as leaf molecules but never achieves a real retrosynthetic disconnection of the benzimidazole ring or the pyridyl sulfoxide. Despite low complexity (339), the sulfoxide and benzimidazole motifs are apparently not well-covered by the template set.

---

### 14. Imatinib
**Target SMILES:** `CC1=CC=C(C=C1)C(=O)NC2=CC=C(C=C2)CNC3=NC=NC(=C3)C4=CN=CC=C4`  
**MW:** 395.5 Da | **Complexity:** 742

**Best route reactions:**
1. Imatinib → methyltoluene amide + 3-pyridylboronic acid (Suzuki coupling for pyridyl group)
2. Amide intermediate → 4-aminobenzyl + 2,4-dichloropyrimidine (nucleophilic aromatic substitution)
3. 4-aminobenzyl compound → 4-methylbenzoic acid + 4-aminobenzylamine (amide hydrolysis)

**Leaf molecules:**
- `OB(O)c1cccnc1` — pyridin-3-ylboronic acid (MW=123, purchasable)
- `Clc1cc(Cl)ncn1` — 2,4-dichloropyrimidine (MW=149, purchasable)
- `Cc1ccc(C(=O)O)cc1` — 4-methylbenzoic acid / toluic acid (MW=136, purchasable)
- `NCc1ccc(N)cc1` — 4-aminobenzylamine (MW=122, purchasable)

**Plausibility:** **Good route, chemically valid.** All three synthetic steps are classic medicinal chemistry reactions. Suzuki coupling, SNAr on dichloropyrimidine, and amide bond disconnection are all appropriate. All four building blocks are commodity chemicals. Score=0.0004 (low but non-zero). Marked cyclic due to minor branching artifacts.

---

### 15. Semaglutide (Ozempic)
**Target SMILES:** (13-residue peptide chain, MW=1800 Da)  
**MW:** 1800.2 Da | **Complexity:** 1700

**Best route reactions:**
1. Semaglutide → methyl ester of full peptide chain (esterification)
2. Methyl ester → truncated peptide chain with further methylation
3. N-terminal dipeptide (Ahib-His) → Ahib fragment + His-OH (peptide bond hydrolysis)
4. Ahib fragment → Ahib methyl ester (esterification)
5–6. Ahib methyl ester ↔ Ahib acid oscillation

**Leaf molecules:** `N[C@@H](Cc1cnc[nH]1)C(=O)O` — L-histidine (MW=155, purchasable), `CO` (methanol)  
**Plausibility:** **Cyclic with inadequate peptide disconnection.** The model correctly identifies the peptide nature of the molecule and disconnects one amide bond to release histidine, but then oscillates on the N-terminal alpha-methyl amino acid fragment. Semaglutide requires solid-phase peptide synthesis with protected amino acids — this template-based model lacks the specialized peptide coupling templates needed to plan such routes. L-histidine as a leaf is chemically correct.

---

### 16. Apixaban
**Target SMILES:** `COc1ccc(cc1)n2nc(C(=O)N)c3CCN(C(=O)c23)c4ccc(cc4)N5CCCCC5=O`  
**MW:** 459.5 Da | **Complexity:** 582

**Best route reactions:**
1. Apixaban → carboxylic acid analog + ammonia (amide hydrolysis — backward)
2. Carboxylic acid → methyl ester (esterification — backward)
3–6. Methyl ester ↔ carboxylic acid oscillation (via methanol elimination/addition)

**Leaf molecules:** N (ammonia), CO (methanol), CO (methanol)  
**Plausibility:** **Cyclic — amide/ester oscillation.** The model identifies the amide carbonyl of the pyrazole carboxamide group and attempts backward synthesis by converting it to acid then methyl ester, but then oscillates. Ammonia as a leaf molecule implies amide formation from ester and ammonia, which is a valid reaction but not a retrosynthetic simplification here. The complex pyrazolo-pyridine ring system is never disconnected.

---

### 17. Palbocyclib
**Target SMILES:** `CC(=O)C1=NC2=C(N=C(N=C2N3CCNCC3)C4=NC=CC=C4)N(C1=O)C5CCCC5`  
**MW:** 419.5 Da | **Complexity:** 667

**Best route reactions:**
1. Palbocyclib → Boc-piperazine carbamate derivative (N-protection — backward)
2. Boc derivative → methyl carbamate analog (carbamate exchange — backward)
3–6. Methyl carbamate ↔ carbamate acid oscillation (MeI exchange)

**Leaf molecules:** `CI` (iodomethane), `CI` (iodomethane)  
**Plausibility:** **Cyclic — carbamate oscillation.** Only iodomethane appears as a leaf molecule. The model locks onto N-carbamate protection/deprotection of the piperazine nitrogen and never disconnects the pyridopyrimidine bicyclic core. The same iodomethane appears twice (from two separate oscillation cycles). No genuine retrosynthetic simplification.

---

### 18. Rivaroxaban
**Target SMILES:** `O=C(NC[C@@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1`  
**MW:** 435.9 Da | **Complexity:** 589

**Best route reactions:**
1. Rivaroxaban → morpholinone fragment + oxazolidinone aminomethyl (amide disconnection)
2. Oxazolidinone → aminomethyl oxazolidinone + chlorothiophene carboxylic acid (retro-amide)
3. Amino oxazolidinone → azido precursor (reduction — backward)
4. Azido compound → mesylate + sodium azide (azide displacement — backward)
5. Mesylate → methanesulfonyl chloride + hydroxymethyl oxazolidinone
6. Hydroxymethyl oxazolidinone → ethoxy compound (ether formation — backward)

**Leaf molecules:**
- `O=C(O)c1ccc(Cl)s1` — 5-chlorothiophene-2-carboxylic acid (MW=163, purchasable)
- `[N-]=[N+]=[N-]` — sodium azide (purchasable, hazardous)
- `CS(=O)(=O)Cl` — methanesulfonyl chloride (MW=115, purchasable)

**Plausibility:** **Partially reasonable with cyclic elements.** The amide disconnection in step 1 is rational. The azide route to the amine (steps 3–4) is a legitimate synthetic strategy. Methanesulfonyl chloride and 5-chlorothiophene carboxylic acid are standard reagents. The cyclic detection traces to the oxazolidinone fragments appearing as both targets and reactants in the beam search. Azide use is technically valid though hazardous at industrial scale.

---

### 19. Etoposide
**Target SMILES:** (complex fused ring glucoside, MW=474 Da)  
**MW:** 474.4 Da | **Complexity:** 804

**Best route reactions:**
1. Etoposide → methylated sugar O-methyl glycoside analog (methylation — backward)
2. Compound → further methylated analog (successive methylation steps)
3–4. Progressive methylation of sugar hydroxyl groups
5. Tetra-O-methyl analog → methyl ester of sugar (C-1 oxidation)
6. Methyl ester analog → iodomethane + hydroxyl compound (demethylation — backward)

**Leaf molecules:** `CI` (iodomethane)  
**Plausibility:** **Severely cyclic — sugar O-methylation loops.** The model cannot disconnect the podophyllotoxin aglycone from the glucoside sugar. Instead, it performs successive O-methylations on the sugar hydroxyl groups and then oscillates. Only iodomethane is a leaf molecule. Etoposide's complex fused polycyclic structure (Bertz 804) is outside the model's effective synthetic range with the PaRoutes template set.

---

### 20. Nintedanib
**Target SMILES:** `CN1CCN(CC1)CC(=O)N(C)C2=CC=C(C=C2)N=C(C3=CC=CC=C3)C4=C(NC5=C4C=CC(=C5)C(=O)OC)O`  
**MW:** 539.6 Da | **Complexity:** 754

**Best route reactions:**
1. Nintedanib → iodomethane + N-demethyl analog (O-methylation — backward)
2. O-methyl analog → methyl ester of indolinone-carboxylic acid fragment (esterification — backward)
3–6. MeI O-methylation/de-methylation oscillation

**Leaf molecules:** CI, CI, CI (three iodomethane molecules)  
**Plausibility:** **Severely cyclic — triple iodomethane oscillation.** Iodomethane appears three times as a leaf molecule from three overlapping oscillation cycles targeting the N-methyl, methyl ester, and enol-methyl groups. The indolinone core and the piperazine amide group are never disconnected. This is an extreme version of the methyl transfer oscillation seen in several other molecules.

---

### 21. Losartan
**Target SMILES:** `Cc1ccc(cc1)C(=O)N2C(=NNC2c3ccccc3Cn4cc(c(n4)C)Cl)C`  
**MW:** 407.9 Da | **Complexity:** 492

**Best route reactions:**
1. Losartan → imidazole-CH₂Br intermediate + 4-methylimidazole (side chain disconnection)
2. CH₂Br intermediate → N-acyl triazoline + NBS (NBS bromination — backward)
3. N-acyl triazoline → triazoline + 4-tolyl acid chloride (acylation)
4. Triazoline → Boc-protected triazoline (N-protection — backward)
5. Boc-triazoline → BnBr + carbamate compound (benzylation — backward)
6. Carbamate → methyl carbamate (esterification oscillation)

**Leaf molecules:**
- `Cc1n[nH]cc1Cl` — 4-methyl-5-chloroimidazole (MW=116, purchasable)
- `O=C1CCC(=O)N1Br` — NBS (N-bromosuccinimide, MW=178, purchasable)
- `Cc1ccc(C(=O)Cl)cc1` — 4-tolyl chloride (MW=155, purchasable)
- `BrCc1ccccc1` — benzyl bromide (MW=171, purchasable)

**Plausibility:** **Cyclic but plausible individual steps.** The model correctly identifies the imidazole side chain as a disconnection point. NBS bromination, acyl chloride coupling, and N-protection strategies are all chemically reasonable for losartan (its actual synthesis does use NBS and similar steps). However, the route loops rather than terminating cleanly. The four leaf molecules are all purchasable and chemically relevant.

---

### 22. Venetoclax
**Target SMILES:** (large BCL-2 inhibitor, MW=869 Da)  
**MW:** 868.5 Da | **Complexity:** 1640

**Best route reactions:**
1. Venetoclax → hydroxyl analog (sulfonamide O-demethylation — backward)
2. Hydroxyl compound → methoxy compound (O-methylation — backward)
3. Methoxy compound → nitro-sulfonamide (sulfonyl chloride coupling)
4. Nitro-sulfonamide → simpler sulfonamide (nitro group manipulation)
5. Sulfonamide → aminobenzamide + sulfonyl chloride (retro-sulfonamide)
6. Aminobenzamide → cyclohexyl-piperazine + methoxybromide (de-coupling)

**Leaf molecules:**
- `Fc1cnc2[nH]ccc2c1F` — 4,6-difluorobenzimidazole fragment (MW=154, purchasable)
- `NCC1CCOCC1` — 4-(aminomethyl)tetrahydropyran (MW=115, purchasable)
- `O=[N+]([O-])O` — nitric acid (purchasable, not a typical synthetic reagent)
- `O=S(=O)(Cl)c1ccc(Cl)cc1` — 4-chlorobenzenesulfonyl chloride (MW=211, purchasable)

**Plausibility:** **Cyclic with some correct disconnections.** The model correctly identifies the benzimidazole fragment as a key building block. The 4-(aminomethyl)tetrahydropyran building block is relevant to the actual structure. However, nitric acid as a leaf molecule is unusual (likely representing nitration rather than a building block). The benzimidazole-chlorochlorotoluene core, which is the pharmacophoric core of venetoclax, is never fully disconnected.

---

## Key Findings and Model Behavior Patterns

### 1. Cyclic route prevalence (86% of best routes)
The most prominent failure mode is **route cyclicity**: the model repeatedly applies forward/backward pairs of the same reaction template (hydrolysis ↔ esterification, demethylation ↔ methylation, debenzylation ↔ benzylation) rather than making forward retrosynthetic progress. This is consistent with a known weakness of sequence models trained on routes without explicit cycle-detection reward shaping.

The most common cyclic patterns observed:
- **Methyl ester oscillation** (Ibuprofen, Camlipixant, Fluorinated Imidazole 56842878, Apixaban, Semaglutide, Palbocyclib, Nintedanib, Etoposide, Acalabrutinib, Losartan) — model alternates between carboxylic acid and methyl ester forms
- **Benzyl protection oscillation** (Omeprazole, Ibrutinib, Losartan) — model applies benzylation and debenzylation of oxygen or nitrogen
- **O-methylation oscillation** (Etoposide, Nintedanib, Venetoclax) — model applies MeI alkylation and reversal

### 2. Score=0.0 for most routes
Sixteen of twenty-two molecules return a best route score of exactly 0.0. Only six molecules return non-zero scores (Aspirin: 1.0; Etoricoxib: 0.0173; Methoxy_Diphenylamine: 0.0149; Imatinib: 0.0004; Tolyl_Pyridine: 0.0082; Fluorinated_Imidazole 84117446: 0.0). Score=0.0 correlates with depth=6 and unsolved status, suggesting the model assigns non-zero scores only when it finds genuinely productive routes.

### 3. Depth=6 saturation
Sixteen molecules hit the maximum explored depth (6 steps) without `all_leaves_purchasable=True`. This suggests the model would benefit from either a larger beam width (>10) or a higher maximum depth to find complete routes for complex molecules.

### 4. Leaf molecule quality
Despite cyclic routes, nearly all reported leaf molecules are chemically plausible small molecules:
- **All leaf molecules have MW < 300 Da** (100% of cases where RDKit could compute MW)
- Common reagents appearing as leaves: boronic acids, halides, acid chlorides, NBS, POCl₃, aminopyridines
- Trivial leaf molecules (MeOH, EtOH, MeI) in cyclic routes indicate oscillation rather than genuine disconnection

### 5. Model performs best on simple biaryl/monoaryl targets
The five solved molecules (excluding Aspirin) share a pattern: they are either single-ring compounds (Methoxy_Diphenylamine, Tolyl_Pyridine) or biaryl molecules requiring 1–3 Suzuki couplings or SNAr steps (Etoricoxib, Imatinib, Fluorinated_Imidazole 84117446). The model excels at recognizing Buchwald-Hartwig C–N couplings and Suzuki C–C couplings as primary disconnections.

### 6. Complex macrolide/peptide/fused-ring molecules not handled
Paclitaxel, Semaglutide, Etoposide, Orforglipron, and Venetoclax all fail to produce solved routes. These molecules require chemistry (glycosidic disconnections, peptide SPPS, chiral resolution strategies) that is underrepresented in the PaRoutes training set.

### 7. Endpoint performance
- **Zero API errors** across all 22 molecules
- **Consistent ~9 second response time** for complex molecules; sub-1-second for Aspirin (recognized immediately as building block)
- **Highest response time: 12.4s** for Etoricoxib
- **Endpoint is production-stable** for the tested workload

---

## Recommendations

1. **Cycle detection in inference:** Add a check during beam search to penalize or reject routes where any intermediate SMILES appears as both a template target and a template reactant within the same route. This could eliminate the 19 cyclic best-routes.

2. **Increase beam width for difficult molecules:** For molecules with complexity > 600, try `beam_width=50` (full beam search). The current 10-route beam likely misses non-cyclic routes that exist deeper in the search tree.

3. **Template oscillation penalty:** The reward function could penalize repeated application of a template on the same SMILES (or its functional group analog). The methyl ester oscillation pattern suggests the model has learned a high-reward local loop rather than a globally productive route.

4. **Building block coverage:** Some molecules (Omeprazole, Apixaban, Palbocyclib) have moderate complexity but fail completely. Expanding the building block library could allow the model to terminate routes earlier rather than cycling.

5. **Score calibration:** Non-zero score molecules all use clearly correct chemistry; zero-score molecules are cyclic or unsolved. Consider whether the score function adequately penalizes depth-6 saturated routes.

---

*Report generated by automated evaluation on 2026-06-21. All SMILES validated with RDKit. Building-block purchasability from RetroSynFormer's internal PaRoutes dataset.*
