"""Molecule-conditioned structured dropout for the Decision Transformer.

Standard dropout uses a single scalar rate applied uniformly to all channels.
Here a small MLP maps the target molecule's Morgan fingerprint to a
*per-channel* drop-probability vector.  The resulting mask is broadcast over
every token in the embedded sequence (returns, states, actions, rewards)
before they enter the GPT-2 encoder, so the transformer must learn
representations that are robust in directions that vary across chemical space.

Integration point
-----------------
`StructuredDropoutDecisionTransformer` wraps `DecisionTransformerModel` and
installs a single forward hook on `embed_ln` — the LayerNorm applied after
all modality embeddings are stacked.  The hook multiplies the normalised
embeddings element-wise by the molecule mask before they reach any
transformer layer.  No other code in the model is touched.

Usage
-----
Enabled by setting ``model.use_structured_dropout: true`` in config.yaml.
Optional knob: ``model.structured_dropout_bottleneck`` (default 128).
"""

import torch
import torch.nn as nn
from transformers import DecisionTransformerConfig, DecisionTransformerModel


class MoleculeConditionedMaskGenerator(nn.Module):
    """MLP from molecule fingerprint → per-channel drop probabilities.

    Architecture: fp_dim → bottleneck (ReLU) → hidden_size (Sigmoid)

    Training: samples a Bernoulli mask and rescales by 1/(1-p) so the
              expected activation magnitude is preserved (inverted dropout).
    Eval:     returns the deterministic expected mask  (1 - p)  instead of
              sampling, for stable, reproducible inference.

    Args:
        fp_dim:      Dimension of the Morgan fingerprint input.
        hidden_size: Number of channels in the DT hidden state (= n_heads × head_dim).
        bottleneck:  Hidden dimension of the MLP (default 128).
    """

    def __init__(self, fp_dim: int, hidden_size: int, bottleneck: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden_size),
            nn.Sigmoid(),
        )
        # Initialise the output bias so the initial drop rate is ~10%
        # sigmoid(-2.2) ≈ 0.10
        nn.init.constant_(self.net[2].bias, -2.2)

    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Return per-channel drop probabilities in (0, 1).

        Args:
            fingerprint: (batch, fp_dim)
        Returns:
            drop_prob:   (batch, hidden_size)

        >>> import torch
        >>> gen = MoleculeConditionedMaskGenerator(fp_dim=8, hidden_size=4, bottleneck=2)
        >>> p = gen(torch.zeros(1, 8))
        >>> p.shape
        torch.Size([1, 4])
        >>> bool((p >= 0).all() and (p <= 1).all())
        True
        >>> p2 = gen(torch.ones(1, 8))
        >>> bool((p != p2).any())
        True
        """
        return self.net(fingerprint)

    def get_mask(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Compute the scale mask to apply to hidden states.

        Returns a tensor of shape (batch, 1, hidden_size) ready to broadcast
        over (batch, n_tokens, hidden_size).

        Training: inverted-dropout Bernoulli mask (stochastic).
        Eval:     deterministic expected-value mask ``(1 - p)``.

        >>> import torch
        >>> gen = MoleculeConditionedMaskGenerator(fp_dim=8, hidden_size=4, bottleneck=2)
        >>> _ = gen.eval()
        >>> mask = gen.get_mask(torch.zeros(1, 8))
        >>> mask.shape
        torch.Size([1, 1, 4])
        >>> bool((mask >= 0).all() and (mask <= 1).all())
        True
        >>> mask2 = gen.get_mask(torch.ones(1, 8))
        >>> bool((mask != mask2).any())
        True
        """
        drop_prob = self(fingerprint).clamp(0.0, 0.9)  # never drop > 90%
        keep_prob = 1.0 - drop_prob                    # (batch, hidden_size)
        if self.training:
            mask = torch.bernoulli(keep_prob) / keep_prob.clamp(min=1e-6)
        else:
            mask = keep_prob
        return mask.unsqueeze(1)  # (batch, 1, hidden_size)


class StructuredDropoutDecisionTransformer(nn.Module):
    """DecisionTransformerModel with molecule-conditioned per-channel dropout.

    Wraps a standard `DecisionTransformerModel` and installs a forward hook on
    `embed_ln` that multiplies every token's embedding by a mask derived from
    the target molecule's fingerprint.  All other behaviour is unchanged.

    Args:
        config:     `DecisionTransformerConfig` (same as for the base model).
        fp_dim:     Fingerprint dimension (config["dataset"]["fp_dim"]).
        bottleneck: Hidden size of the mask-generator MLP (default 128).
    """

    def __init__(
        self,
        config: DecisionTransformerConfig,
        fp_dim: int,
        bottleneck: int = 128,
    ):
        super().__init__()
        self.dt = DecisionTransformerModel(config)
        self.mask_gen = MoleculeConditionedMaskGenerator(
            fp_dim, config.hidden_size, bottleneck
        )
        self._mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Expose the same attribute surface as the wrapped model so that
    # existing code (e.g. torch.save(model.state_dict())) keeps working.
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.dt, name)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _hook(self, module, input, output):
        """Applied to embed_ln output: multiply by molecule mask."""
        if self._mask is not None:
            return output * self._mask
        return output

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        # Extract target molecule fingerprint: first position, first fp_dim features.
        # states: (batch, seq_len, state_dim)  where state_dim = fp_dim * n_in_state
        fp_dim = self.mask_gen.net[0].in_features
        molecule_fp = states[:, 0, :fp_dim].to(states.dtype)

        # Compute mask and stash it so the hook can read it.
        self._mask = self.mask_gen.get_mask(molecule_fp)  # (batch, 1, hidden_size)

        handle = self.dt.embed_ln.register_forward_hook(self._hook)
        try:
            out = self.dt(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
        finally:
            handle.remove()
            self._mask = None

        return out

    # ------------------------------------------------------------------
    # Serialisation helpers: save/load as a single state_dict
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        config: DecisionTransformerConfig,
        fp_dim: int,
        bottleneck: int = 128,
        map_location=None,
    ) -> "StructuredDropoutDecisionTransformer":
        model = cls(config, fp_dim, bottleneck)
        model.load_state_dict(torch.load(path, map_location=map_location))
        return model
