"""Tests for molecule-conditioned structured dropout.

Run with -s to see the full statistics tables:
    pytest tests/test_structured_dropout.py -v -s
"""

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from retrosynformer.structured_dropout import MoleculeConditionedMaskGenerator
from retrosynformer.utils.utils import get_morgan_fingerprint

FP_DIM = 1024
HIDDEN_SIZE = 256
BOTTLENECK = 128
RADIUS = 2

# Chemically diverse panel spanning extreme molecular properties:
#   size, aromaticity, polarity, heteroatom content, ring complexity.
MOLECULES = {
    "methane":      "C",                                        # MW=16,  no rings, no aromatic, non-polar
    "octane":       "CCCCCCCC",                                 # MW=114, no rings, aliphatic chain
    "benzene":      "c1ccccc1",                                 # MW=78,  1 aromatic ring, non-polar
    "naphthalene":  "c1ccc2ccccc2c1",                          # MW=128, 2 fused aromatic rings
    "glucose":      "OCC1OC(O)C(O)C(O)C1O",                   # MW=180, 1 ring, highly polar (4 OH)
    "aspirin":      "CC(=O)Oc1ccccc1C(=O)O",                  # MW=180, aromatic + ester + acid
    "ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",             # MW=206, aromatic, non-polar core
    "caffeine":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",             # MW=194, N-heteroaromatic, rigid
    "cholesterol":  "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",  # MW=386, 4 rings, steroid
}


@pytest.fixture(scope="module")
def mask_gen():
    torch.manual_seed(42)
    gen = MoleculeConditionedMaskGenerator(FP_DIM, HIDDEN_SIZE, BOTTLENECK)
    gen.eval()
    return gen


@pytest.fixture(scope="module")
def fingerprints():
    return {
        name: get_morgan_fingerprint(smi, radius=RADIUS, n_bits=FP_DIM).float().unsqueeze(0)
        for name, smi in MOLECULES.items()
    }


@pytest.fixture(scope="module")
def drop_probs(mask_gen, fingerprints):
    """Per-molecule drop-probability vectors, shape (HIDDEN_SIZE,)."""
    probs = {}
    with torch.no_grad():
        for name, fp in fingerprints.items():
            probs[name] = mask_gen(fp).squeeze(0).numpy()
    return probs


# ---------------------------------------------------------------------------
# Hard assertions
# ---------------------------------------------------------------------------

def test_different_molecules_produce_different_masks(drop_probs):
    """Every molecule pair must yield a distinct drop-probability vector."""
    names = list(drop_probs)
    vecs = np.stack([drop_probs[n] for n in names])
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(vecs[i], vecs[j], atol=1e-6), (
                f"Molecules '{names[i]}' and '{names[j]}' produced identical masks"
            )


def test_mask_values_in_valid_range(mask_gen, fingerprints):
    """Eval-mode masks (1-p) must be in [0, 1]."""
    with torch.no_grad():
        for name, fp in fingerprints.items():
            mask = mask_gen.get_mask(fp).squeeze()
            assert mask.min() >= 0.0, f"{name}: mask has values < 0"
            assert mask.max() <= 1.0, f"{name}: mask has values > 1"


def test_eval_mask_is_deterministic(mask_gen, fingerprints):
    """Two consecutive eval calls on the same molecule must give identical masks."""
    fp = fingerprints["aspirin"]
    with torch.no_grad():
        m1 = mask_gen.get_mask(fp).numpy()
        m2 = mask_gen.get_mask(fp).numpy()
    assert np.allclose(m1, m2), "Eval mask is not deterministic"


def test_train_mask_is_stochastic(mask_gen, fingerprints):
    """Training-mode Bernoulli masks must differ between calls (with high probability)."""
    fp = fingerprints["aspirin"]
    mask_gen.train()
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            m1 = mask_gen.get_mask(fp).numpy()
            torch.manual_seed(1)
            m2 = mask_gen.get_mask(fp).numpy()
        assert not np.allclose(m1, m2), "Training masks should be stochastic but are identical"
    finally:
        mask_gen.eval()


def test_initial_drop_rate_near_ten_percent(drop_probs):
    """Bias init of -2.2 should give ~10% mean drop probability (sigmoid(-2.2) ≈ 0.10)."""
    for name, p in drop_probs.items():
        mean_drop = p.mean()
        assert 0.05 < mean_drop < 0.20, (
            f"{name}: mean drop probability {mean_drop:.4f} far from expected ~0.10"
        )


def test_channels_have_spatial_variation(drop_probs):
    """Drop probabilities must vary across embedding channels (not uniform over HIDDEN_SIZE)."""
    for name, p in drop_probs.items():
        assert p.std() > 1e-4, (
            f"{name}: mask has no spatial variation (std={p.std():.6f}); "
            "hidden channels all have the same drop probability"
        )


def test_mask_cross_molecule_channel_variation(drop_probs):
    """Each channel's drop probability must vary across molecules (mask responds to chemistry)."""
    matrix = np.stack(list(drop_probs.values()))  # (n_mols, HIDDEN_SIZE)
    per_channel_std = matrix.std(axis=0)          # (HIDDEN_SIZE,)
    assert per_channel_std.mean() > 1e-4, (
        "All channels respond identically to all molecules; structured dropout is not molecule-conditioned"
    )


# ---------------------------------------------------------------------------
# Statistics (visible with pytest -s)
# ---------------------------------------------------------------------------

def test_dropout_distribution_statistics(drop_probs):
    """Print dropout statistics across molecules and channels.

    Columns shown per molecule:
      mean_drop — average drop probability over all HIDDEN_SIZE channels
      std       — std of drop probs over channels (spatial spread)
      p25/p75   — quartiles

    Per-channel summary shows which channels are most molecule-sensitive.
    Pairwise cosine similarity shows how distinct each molecule's mask is.
    Property correlations show (at random-init, expect ~0) which RDKit properties
    predict mean dropout; after training these should reflect chemically meaningful
    associations between molecule features and embedding channel reliability.
    """
    names = list(drop_probs)
    matrix = np.stack([drop_probs[n] for n in names])  # (n_mols, HIDDEN_SIZE)

    # -- per-molecule RDKit properties --
    mol_props = {}
    for name in names:
        mol = Chem.MolFromSmiles(MOLECULES[name])
        mol_props[name] = {
            "MW":        Descriptors.MolWt(mol),
            "ArRings":   Descriptors.NumAromaticRings(mol),
            "RingCount": Descriptors.RingCount(mol),
            "HBA":       Descriptors.NumHAcceptors(mol),
            "HBD":       Descriptors.NumHDonors(mol),
            "TPSA":      Descriptors.TPSA(mol),
            "RotBonds":  Descriptors.NumRotatableBonds(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
        }

    # -- per-molecule dropout summary --
    print("\n" + "=" * 80)
    print(f"DROP PROBABILITY per molecule  (HIDDEN_SIZE={HIDDEN_SIZE} channels, BOTTLENECK={BOTTLENECK})")
    print(f"{'molecule':<14} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} "
          f"{'p25':>8} {'p75':>8}  {'MW':>6} {'ArRings':>7}")
    print("-" * 80)
    for name in names:
        v = drop_probs[name]
        p = mol_props[name]
        print(f"{name:<14} {v.mean():8.4f} {v.std():8.4f} {v.min():8.4f} {v.max():8.4f} "
              f"{np.percentile(v, 25):8.4f} {np.percentile(v, 75):8.4f}  "
              f"{p['MW']:6.1f} {p['ArRings']:7d}")

    # -- per-channel summary --
    ch_mean  = matrix.mean(axis=0)
    ch_std   = matrix.std(axis=0)
    ch_range = matrix.max(axis=0) - matrix.min(axis=0)

    print("\n" + "=" * 80)
    print("PER-CHANNEL statistics (across all molecules):")
    print(f"  channel mean drop-prob  — mean={ch_mean.mean():.4f}  std={ch_mean.std():.4f}  "
          f"min={ch_mean.min():.4f}  max={ch_mean.max():.4f}")
    print(f"  channel std  drop-prob  — mean={ch_std.mean():.4f}  std={ch_std.std():.4f}  "
          f"min={ch_std.min():.4f}  max={ch_std.max():.4f}")
    print(f"  channel range drop-prob — mean={ch_range.mean():.4f}  std={ch_range.std():.4f}  "
          f"min={ch_range.min():.4f}  max={ch_range.max():.4f}")
    print(f"  fraction of channels with range > 0.05: "
          f"{(ch_range > 0.05).mean():.3f}")

    top_k = 10
    top_idx = np.argsort(ch_std)[::-1][:top_k]
    print(f"\n  Top {top_k} most molecule-sensitive channels (highest cross-molecule std):")
    print(f"  {'ch':>5} {'mean':>8} {'std':>8} {'range':>8}")
    for i in top_idx:
        print(f"  {i:>5} {ch_mean[i]:8.4f} {ch_std[i]:8.4f} {ch_range[i]:8.4f}")

    # -- pairwise cosine similarity --
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed = matrix / norms
    cosine_sim = normed @ normed.T
    tril_i, tril_j = np.tril_indices(len(names), k=-1)
    pair_sims = cosine_sim[tril_i, tril_j]

    print("\n" + "=" * 80)
    print("PAIRWISE COSINE SIMILARITY between drop-prob vectors:")
    print(f"  mean={pair_sims.mean():.4f}  std={pair_sims.std():.4f}  "
          f"min={pair_sims.min():.4f}  max={pair_sims.max():.4f}")
    sorted_k = np.argsort(pair_sims)
    print("  Most similar (masks most alike — structurally related molecules):")
    for k in sorted_k[-3:][::-1]:
        a, b = names[tril_i[k]], names[tril_j[k]]
        print(f"    {a:<14} ↔ {b:<14}  sim={pair_sims[k]:.4f}")
    print("  Most dissimilar (masks most different):")
    for k in sorted_k[:3]:
        a, b = names[tril_i[k]], names[tril_j[k]]
        print(f"    {a:<14} ↔ {b:<14}  sim={pair_sims[k]:.4f}")

    # -- property correlation with mean drop rate --
    mol_mean_drop = matrix.mean(axis=1)  # (n_mols,)
    prop_names = list(next(iter(mol_props.values())))

    print("\n" + "=" * 80)
    print("PROPERTY CORRELATION with mean drop probability")
    print("  (random-init: expect ~0; after training: non-zero = chemistry drives masking)")
    print(f"  {'property':<12} {'Pearson r':>10}  interpretation")
    print("  " + "-" * 55)
    for prop in prop_names:
        vals = np.array([mol_props[n][prop] for n in names])
        r = np.corrcoef(vals, mol_mean_drop)[0, 1]
        if r > 0.4:
            note = "↑ higher drop with larger/more"
        elif r < -0.4:
            note = "↓ lower drop with larger/more"
        else:
            note = "no clear trend at this init"
        print(f"  {prop:<12} {r:>10.4f}  {note}")

    print("=" * 80 + "\n")

    # soft assertions — must hold even at random init
    assert ch_std.mean() > 0, "Cross-molecule channel std is zero"
    assert pair_sims.max() < 1.0 - 1e-6, "Some molecule pair has numerically identical masks"
    assert ch_range.max() > 0, "No channel varies across any molecules"
