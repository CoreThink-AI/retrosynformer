"""Parameter counts and weight-complexity/entropy estimates for a trained model.

Operates on a plain ``state_dict`` (as saved by ``trainer.py`` via
``torch.save(model.state_dict(), ...)`` — see ``compression.load_model`` for
reading one back off disk) rather than a live ``nn.Module``, so it works
identically on a checkpoint that was never actually loaded into a model.
Nothing here is specific to RetroSynFormer's Decision Transformer, or to
generative/transformer models in general — the same functions apply to any
PyTorch ``state_dict``: CNNs, RNNs, plain MLPs, encoder-only classifiers,
diffusion U-Nets, etc. Categorisation falls back to shape-based buckets
(``convolution``/``weight_matrix``/``vector``/``scalar``) whenever a
tensor's name doesn't match a recognised naming convention, so unfamiliar
architectures still get a sensible (if less granular) breakdown instead of
an error.

Four complementary complexity/entropy estimates are computed per weight
tensor, because no single number captures "how complex is this matrix":

- **Spectral (effective rank / stable rank)** — how many independent
  directions the matrix's energy is spread across. A rank-collapsed
  attention or MLP matrix (common in over-parameterised or under-trained
  transformers) has a low effective rank relative to its shape.
- **Value-distribution entropy (histogram)** — how uniformly the raw weight
  *values* are spread, independent of matrix shape/rank. Works on any
  tensor, including 1-D biases and layernorm scales.
- **Gaussian differential entropy** — a cheap closed-form entropy estimate
  assuming the weight values are normally distributed (a good approximation
  for most trained NN weights), commonly used in MDL/quantization analyses.
- **Whole-file compression ratio** — a Kolmogorov-complexity proxy that
  captures cross-tensor redundancy the other three (which look at one
  tensor's value distribution or spectrum in isolation) cannot see.

None of these require a forward pass or the training dataset — they are
static analyses of the weight tensors themselves, not of the model's
activations or attention patterns on real inputs.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Naming conventions — best-effort and architecture-agnostic
# ---------------------------------------------------------------------------

# Substrings are matched against the lowercased name and checked in order,
# so more specific categories (attention) are tried before generic ones
# (feedforward) that could otherwise shadow them (e.g. "self_attn.fc" would
# match both "attn" and "fc"). These cover common PyTorch naming conventions
# across transformers (HF GPT-2/BERT/ViT style), CNNs (torchvision ResNet/
# VGG style), and RNNs — not any one model family.
_NAME_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gating",         ("mask_generator", "gate")),
    ("attention",      ("attn", "attention")),
    ("convolution",    ("conv",)),
    ("recurrent",      ("lstm", "gru", "rnn", "weight_ih", "weight_hh")),
    ("feedforward",    ("mlp", "ffn", "feedforward", "fc", "dense", "linear", "intermediate")),
    ("normalization",  ("norm", "ln_", "bn", "batchnorm", "layernorm", "groupnorm")),
    ("embedding",      ("embed", "wte", "wpe", "emb")),
    ("head",           ("predict", "classifier", "logits", "readout", "head", "output_proj")),
    ("bias",           ("bias",)),
)


def classify_param_name(name: str, tensor: "torch.Tensor | None" = None) -> str:
    """Best-effort architectural category for a state_dict entry.

    Matches common cross-architecture naming substrings (see
    ``_NAME_CATEGORY_RULES``) rather than any one model's module names, so
    it works reasonably on transformers, CNNs, RNNs, and plain MLPs alike.
    When nothing matches and *tensor* is given, falls back to a purely
    shape-based category so every tensor still lands somewhere sensible:
    ``convolution`` (4+ dims), ``weight_matrix`` (2-D), ``vector`` (1-D), or
    ``scalar`` (0-D). Without *tensor*, unmatched names return ``"other"``.
    """
    lname = name.lower()
    for category, substrings in _NAME_CATEGORY_RULES:
        if any(s in lname for s in substrings):
            return category

    if tensor is not None:
        dim = tensor.dim()
        if dim >= 4:
            return "convolution"
        if dim == 2:
            return "weight_matrix"
        if dim == 1:
            return "vector"
        return "scalar"
    return "other"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(state_dict: dict) -> dict:
    """Return total / per-category parameter counts for *state_dict*.

    Returns
    -------
    dict with keys:
        total_parameters:          sum of numel() across every tensor
        total_tensors:             number of tensor entries
        dtype:                     dtype of the first floating-point tensor found
        weight_matrix_parameters:  parameters in 2-D+ tensors whose leaf name
                                    contains "weight" (the ones spectral_stats
                                    can analyse — covers ".weight" and RNN-style
                                    "weight_ih_l0"/"weight_hh_l0" alike)
        bias_and_norm_parameters:  everything else (biases, 1-D layernorm/batchnorm scales)
        by_category:               {category: {"parameters": int, "tensors": int}}
    """
    by_category: dict[str, dict[str, int]] = {}
    total_parameters = 0
    total_tensors = 0
    weight_matrix_parameters = 0
    dtype: Optional[str] = None

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        n = int(tensor.numel())
        total_parameters += n
        total_tensors += 1
        if dtype is None and torch.is_floating_point(tensor):
            dtype = str(tensor.dtype)

        cat = classify_param_name(name, tensor)
        bucket = by_category.setdefault(cat, {"parameters": 0, "tensors": 0})
        bucket["parameters"] += n
        bucket["tensors"] += 1

        # "weight" appears in the leaf name for almost every PyTorch module's
        # main 2-D+ parameter — nn.Linear/Conv/Embedding use ".weight", RNNs
        # use "weight_ih_l0"/"weight_hh_l0" — so this check is name-convention
        # based rather than tied to any one architecture.
        leaf = name.rsplit(".", 1)[-1]
        if "weight" in leaf and tensor.dim() >= 2:
            weight_matrix_parameters += n

    return {
        "total_parameters": total_parameters,
        "total_tensors": total_tensors,
        "dtype": dtype,
        "weight_matrix_parameters": weight_matrix_parameters,
        "bias_and_norm_parameters": total_parameters - weight_matrix_parameters,
        "by_category": by_category,
    }


# ---------------------------------------------------------------------------
# Per-tensor complexity metrics
# ---------------------------------------------------------------------------

def spectral_stats(weight: torch.Tensor) -> dict:
    """Singular-value-based complexity metrics for a weight tensor.

    Tensors with more than 2 dimensions (e.g. a Conv2d kernel shaped
    ``[out_channels, in_channels, kh, kw]``) are reshaped to 2-D by flattening
    everything after the first axis — the standard way to expose a
    convolution kernel's rank structure (``out_channels x (in_channels*kh*kw)``).
    For a plain Linear/attention-projection weight this reshape is a no-op.

    - ``effective_rank``: ``exp(H)`` where ``H`` is the Shannon entropy
      (nats) of the L1-normalised singular-value spectrum (Roy & Vetterli,
      2007, "The effective rank: A measure of effective dimensionality").
      Ranges from 1 (all energy along one direction — a maximally simple,
      near-rank-1 matrix) to ``min(shape)`` (energy spread evenly across
      every direction — maximally complex for its size).
    - ``stable_rank``: ``||W||_F^2 / sigma_max^2`` — a cheaper, less
      outlier-sensitive proxy for the same idea.
    - ``spectral_entropy_bits``: the same Shannon entropy in bits.
    - ``condition_number``: ``sigma_max / sigma_min`` (``inf`` if the matrix
      is exactly rank-deficient).
    - ``spectral_norm`` / ``frobenius_norm``: ``sigma_max`` and ``||W||_F``.
    - ``rank_dim``: ``min(shape)`` after reshaping — the maximum rank the
      matrix could have, for normalising ``effective_rank`` across
      differently-shaped tensors.

    Raises ``ValueError`` if *weight* has fewer than 2 dimensions.
    """
    if weight.dim() < 2:
        raise ValueError(
            f"spectral_stats requires a tensor with >= 2 dims, got shape {tuple(weight.shape)}"
        )

    w = weight.detach().to(torch.float32)
    if w.dim() > 2:
        w = w.reshape(w.shape[0], -1)
    s = torch.linalg.svdvals(w)
    s = s[s > 0]  # exact zeros would make log(0) undefined when normalising below
    if s.numel() == 0:
        return {
            "effective_rank": 0.0, "stable_rank": 0.0, "spectral_entropy_bits": 0.0,
            "condition_number": float("inf"), "spectral_norm": 0.0, "frobenius_norm": 0.0,
            "rank_dim": min(w.shape),
        }

    p = s / s.sum()
    entropy_nats = -(p * p.log()).sum().item()
    frob_sq = (s ** 2).sum().item()
    sigma_max = s[0].item()
    sigma_min = s[-1].item()

    return {
        "effective_rank": math.exp(entropy_nats),
        "stable_rank": frob_sq / (sigma_max ** 2) if sigma_max > 0 else 0.0,
        "spectral_entropy_bits": entropy_nats / math.log(2),
        "condition_number": (sigma_max / sigma_min) if sigma_min > 0 else float("inf"),
        "spectral_norm": sigma_max,
        "frobenius_norm": math.sqrt(frob_sq),
        "rank_dim": min(w.shape),
    }


def histogram_entropy_bits(tensor: torch.Tensor, bins: int = 256) -> float:
    """Shannon entropy (bits) of *tensor*'s value histogram.

    Treats the flattened tensor as draws from an unknown distribution, bins
    the values into *bins* equal-width buckets, and computes
    ``-sum(p * log2(p))`` over the empirical (non-empty) bucket frequencies.
    Unlike :func:`spectral_stats`, this needs no particular shape or
    dimensionality, so it applies equally to weight matrices, biases, and
    layernorm scales. Returns 0.0 for an empty or constant tensor (both
    carry no information under this measure).
    """
    values = tensor.detach().to(torch.float32).flatten()
    if values.numel() == 0:
        return 0.0
    vmin, vmax = values.min().item(), values.max().item()
    if vmin == vmax:
        return 0.0
    hist = torch.histc(values, bins=bins, min=vmin, max=vmax)
    total = hist.sum().item()
    p = hist[hist > 0] / total
    return -(p * p.log2()).sum().item()


def gaussian_differential_entropy_bits(tensor: torch.Tensor) -> float:
    """Per-weight differential entropy (bits), assuming *tensor* is Gaussian.

    ``H = 0.5 * log2(2 * pi * e * sigma^2)`` for ``sigma = tensor.std()`` —
    a closed-form, binning-free entropy estimate widely used for weight and
    activation entropy in NN quantization and MDL-based pruning analyses. It
    is exact when the weight distribution truly is Gaussian, which is a
    reasonable approximation for most trained NN weight tensors (unimodal,
    roughly bell-shaped). Unlike discrete (Shannon) entropy, *differential*
    entropy can be negative — that just means the distribution is narrower
    than a unit-variance reference, not an error.

    Returns ``-inf`` for a constant or single-element tensor (``std`` is 0 or
    undefined), which carries no information under any continuous-entropy
    measure.
    """
    flat = tensor.detach().to(torch.float32).flatten()
    if flat.numel() < 2:
        return float("-inf")  # unbiased std of < 2 samples is 0 or undefined
    std = flat.std().item()
    if std <= 0:
        return float("-inf")
    return 0.5 * math.log2(2 * math.pi * math.e * std ** 2)


def compression_ratio_bits_per_weight(
    path: "str | Path", *, codec: str = "bz2", total_parameters: Optional[int] = None,
) -> dict:
    """Whole-file lossless-compression estimate of *path*'s information content.

    Compresses *path* (a model checkpoint file already on disk) with *codec*
    and reports the compressed/uncompressed byte ratio. This is a
    Kolmogorov-complexity proxy: a highly redundant checkpoint (near-duplicate
    rows, aggressive quantization, strong low-rank structure) compresses
    well; a "high entropy" one does not. It complements the per-tensor
    spectral/histogram/Gaussian estimates, which look at one tensor's value
    distribution or spectrum in isolation and cannot see redundancy *across*
    tensors (e.g. weight tying, structured pruning masks).

    Pass *total_parameters* (e.g. from :func:`count_parameters`) to also get
    ``bits_per_parameter`` — the compressed size expressed as an average bit
    budget per learned parameter, comparable across model sizes.
    """
    from retrosynformer.compression import codec_extension, compress_file

    path = Path(path)
    original_bytes = path.stat().st_size
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / (path.name + codec_extension(codec))
        compress_file(path, dst, codec)
        compressed_bytes = dst.stat().st_size

    result = {
        "codec": codec,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compressed_bytes / original_bytes if original_bytes else 0.0,
    }
    if total_parameters:
        result["bits_per_parameter"] = compressed_bytes * 8 / total_parameters
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _weighted_mean(key: str, entries: list[dict]) -> Optional[float]:
    """Parameter-count-weighted mean of *key* across *entries*, skipping non-finite values."""
    pairs = [(e["parameters"], e[key]) for e in entries if key in e and math.isfinite(e[key])]
    total_w = sum(w for w, _ in pairs)
    return sum(w * v for w, v in pairs) / total_w if total_w else None


def weight_complexity_report(
    state_dict: dict,
    *,
    model_path: "str | Path | None" = None,
    codec: str = "bz2",
    histogram_bins: int = 256,
    include_per_tensor: bool = False,
) -> dict:
    """Compute parameter counts + entropy/complexity estimates for *state_dict*.

    Every floating-point tensor gets histogram and Gaussian entropy
    estimates; every tensor with 2 or more dimensions (linear/attention
    weights, conv kernels, embedding tables, ...) additionally gets
    :func:`spectral_stats`. Results are aggregated by
    :func:`classify_param_name` category and overall (both parameter-count
    weighted, so a handful of huge weight matrices don't get drowned out by
    hundreds of tiny biases or normalization scales, and vice versa).

    Parameters
    ----------
    state_dict:
        A model's state dict, as returned by ``compression.load_model`` or
        ``model.state_dict()``.
    model_path:
        Path to the checkpoint file on disk. If given, also computes
        :func:`compression_ratio_bits_per_weight` and includes it under
        ``overall.compression``. Omit to skip the (file-I/O) compression step.
    codec:
        Compression codec for the file-based estimate: "gz", "bz2" (default),
        or "xz". See :mod:`retrosynformer.compression`.
    histogram_bins:
        Bin count for :func:`histogram_entropy_bits`.
    include_per_tensor:
        If True, include a ``"per_tensor"`` list with one entry per analysed
        tensor (name, category, shape, dtype, and all applicable metrics).
        Off by default since a 24-layer model has ~300 tensors.

    Returns
    -------
    dict with keys:
        parameters:   output of :func:`count_parameters`
        by_category:  {category: aggregated stats + tensor/parameter counts}
        overall:      aggregated stats across every analysed tensor, plus
                      ``compression`` (if *model_path* given)
        per_tensor:   (only if *include_per_tensor*) full per-tensor list
    """
    params = count_parameters(state_dict)

    per_tensor: list[dict] = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
            continue
        entry = {
            "name": name,
            "category": classify_param_name(name, tensor),
            "shape": [int(d) for d in tensor.shape],
            "dtype": str(tensor.dtype),
            "parameters": int(tensor.numel()),
            "histogram_entropy_bits": histogram_entropy_bits(tensor, bins=histogram_bins),
            "gaussian_entropy_bits": gaussian_differential_entropy_bits(tensor),
        }
        if tensor.dim() >= 2:
            entry.update(spectral_stats(tensor))
        per_tensor.append(entry)

    by_category: dict[str, dict] = {}
    for cat in sorted({e["category"] for e in per_tensor}):
        entries = [e for e in per_tensor if e["category"] == cat]
        by_category[cat] = {
            "tensors": len(entries),
            "parameters": sum(e["parameters"] for e in entries),
            "mean_histogram_entropy_bits": _weighted_mean("histogram_entropy_bits", entries),
            "mean_gaussian_entropy_bits": _weighted_mean("gaussian_entropy_bits", entries),
            "mean_effective_rank": _weighted_mean("effective_rank", entries),
            "mean_stable_rank": _weighted_mean("stable_rank", entries),
        }

    weight_matrices = [e for e in per_tensor if "effective_rank" in e]
    overall = {
        "n_tensors_analyzed": len(per_tensor),
        "n_weight_matrices_analyzed": len(weight_matrices),
        "mean_histogram_entropy_bits": _weighted_mean("histogram_entropy_bits", per_tensor),
        "mean_gaussian_entropy_bits": _weighted_mean("gaussian_entropy_bits", per_tensor),
        "mean_effective_rank": _weighted_mean("effective_rank", weight_matrices),
        "mean_stable_rank": _weighted_mean("stable_rank", weight_matrices),
    }
    if model_path is not None:
        overall["compression"] = compression_ratio_bits_per_weight(
            model_path, codec=codec, total_parameters=params["total_parameters"],
        )

    report = {"parameters": params, "by_category": by_category, "overall": overall}
    if include_per_tensor:
        report["per_tensor"] = per_tensor
    return report
