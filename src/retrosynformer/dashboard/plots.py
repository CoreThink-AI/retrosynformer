"""Plotly figure builders shared by the dashboard and rs-plot-learning-curves.

Single-trial learning curves are built here directly with plotly.
Study-level analysis (parallel coordinates, optimization history, param
importances) delegates to optuna.visualization, which returns plotly Figures.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ORANGE = "#fd7e14"
_BLUE   = "#007bff"
_GRAY   = "#6c757d"


def build_trial_figure(epochs_data: list[dict], title: str = "") -> go.Figure:
    """Return a 2×2 subplot Plotly figure for one trial's learning curves."""
    if not epochs_data:
        fig = go.Figure()
        fig.update_layout(title=title or "No epoch data yet", template="plotly_white")
        return fig

    epochs = [r.get("epoch", i) for i, r in enumerate(epochs_data)]

    def _vals(key: str) -> list:
        return [r.get(key) for r in epochs_data]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss", "Action Accuracy", "Route Accuracy", "Seconds / Epoch"),
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=_vals("train_loss"), name="train",
                             line=dict(color=_ORANGE, width=1.5), mode="lines",
                             legendgroup="train"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=_vals("valid_loss"), name="valid",
                             line=dict(color=_BLUE, width=1.5), mode="lines",
                             legendgroup="valid"), row=1, col=1)

    # Action Accuracy
    fig.add_trace(go.Scatter(x=epochs, y=_vals("train_action_accuracy"), name="train",
                             line=dict(color=_ORANGE, width=1.5), mode="lines",
                             showlegend=False, legendgroup="train"), row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=_vals("valid_action_accuracy"), name="valid",
                             line=dict(color=_BLUE, width=1.5), mode="lines",
                             showlegend=False, legendgroup="valid"), row=1, col=2)

    # Route Accuracy
    fig.add_trace(go.Scatter(x=epochs, y=_vals("train_route_accuracy"), name="train",
                             line=dict(color=_ORANGE, width=1.5), mode="lines",
                             showlegend=False, legendgroup="train"), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=_vals("valid_route_accuracy"), name="valid",
                             line=dict(color=_BLUE, width=1.5), mode="lines",
                             showlegend=False, legendgroup="valid"), row=2, col=1)

    # Seconds / Epoch
    fig.add_trace(go.Scatter(x=epochs, y=_vals("seconds_per_epoch"), name="sec/epoch",
                             line=dict(color=_GRAY, width=1.5), mode="lines",
                             showlegend=False), row=2, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=600,
        margin=dict(l=50, r=20, t=80, b=50),
        legend=dict(orientation="h", yanchor="top", y=-0.04, x=0.5, xanchor="center"),
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Epoch")

    return fig


# ---------------------------------------------------------------------------
# Study-level figures — delegate to optuna.visualization
# ---------------------------------------------------------------------------

# Preferred parameter order for the parallel coordinates axes.
_PARCOORDS_PARAM_ORDER = [
    "n_heads", "n_layers", "head_dim", "hidden_size",
    "lr", "attn_pdrop", "embd_pdrop", "resid_pdrop",
    "structured_dropout_rate", "structured_dropout_bottleneck",
]


def load_optuna_study(db_path: str, study_name: str):
    """Load an optuna.Study from a SQLite file (read-only, no side effects)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )


def build_parcoords_figure(optuna_study) -> go.Figure:
    """Scatter matrix (SPLOM) coloured by objective value.

    go.Parcoords requires WebGL; go.Splom is pure SVG and works everywhere.
    Each cell shows the pairwise scatter for two hyperparameters; yellow dots
    = high objective score, purple = low.  Hover shows trial number and value.
    """
    complete = [t for t in optuna_study.trials
                if t.state.name == "COMPLETE" and t.value is not None]
    if not complete:
        fig = go.Figure()
        fig.update_layout(title="No complete trials yet", template="plotly_white")
        return fig

    sampled: set[str] = set()
    for t in complete:
        sampled.update(t.params.keys())

    ordered = [p for p in _PARCOORDS_PARAM_ORDER if p in sampled]
    rest    = sorted(sampled - set(ordered))
    param_names = ordered + rest

    dimensions = []
    for p in param_names:
        values = [t.params.get(p) for t in complete]
        if all(v is None for v in values):
            continue
        dimensions.append(dict(label=p, values=values))

    scores = [t.value for t in complete]
    # Append the objective as the final column so score vs each param is visible.
    dimensions.append(dict(label="objective", values=scores))

    fig = go.Figure(go.Splom(
        dimensions=dimensions,
        marker=dict(
            color=scores,
            colorscale="Viridis",
            showscale=True,
            size=7,
            line=dict(width=0.3, color="white"),
            colorbar=dict(title="objective", thickness=14),
        ),
        text=[f"trial {t.number}: {t.value:.4f}" for t in complete],
        diagonal_visible=True,
        showupperhalf=False,
    ))

    n = len(dimensions)
    fig.update_layout(
        title="Hyperparameter scatter matrix",
        height=max(500, 140 * n),
        margin=dict(l=60, r=20, t=60, b=60),
        template="plotly_white",
    )
    return fig


def build_optimization_history_figure(optuna_study) -> go.Figure:
    """Objective value vs trial number, with running best highlighted."""
    import optuna.visualization as ov

    fig = ov.plot_optimization_history(optuna_study)
    fig.update_layout(
        title="Optimization history",
        height=380,
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return fig


def build_param_importances_figure(optuna_study) -> go.Figure | None:
    """Hyperparameter importances (fANOVA).  Returns None if < 4 complete trials."""
    import optuna.visualization as ov

    complete = [t for t in optuna_study.trials
                if t.state.name == "COMPLETE" and t.value is not None]
    if len(complete) < 4:
        return None

    fig = ov.plot_param_importances(optuna_study)
    fig.update_layout(
        title="Hyperparameter importances (fANOVA)",
        height=380,
        margin=dict(l=160, r=30, t=50, b=50),
    )
    return fig
