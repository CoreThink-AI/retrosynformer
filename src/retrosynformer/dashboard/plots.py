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
    """Lower-triangle scatter matrix coloured by objective value.

    Uses individual go.Scatter traces (pure SVG) arranged in a subplot grid,
    avoiding go.Parcoords and go.Splom which both require WebGL.
    Yellow dots = high objective score, purple = low.
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
    dims    = ordered + rest + ["objective"]

    pv = {p: [t.params.get(p) for t in complete] for p in dims[:-1]}
    pv["objective"] = [t.value for t in complete]
    scores = pv["objective"]
    hover  = [f"trial {t.number}: {t.value:.4f}" for t in complete]

    n   = len(dims)
    fig = make_subplots(rows=n, cols=n,
                        horizontal_spacing=0.025, vertical_spacing=0.025)

    first = True
    for ri in range(n):
        for ci in range(n):
            r, c = ri + 1, ci + 1
            if ci < ri:
                cb = dict(title="obj", thickness=10, len=0.4, x=1.02) if first else {}
                fig.add_trace(
                    go.Scatter(
                        x=pv[dims[ci]], y=pv[dims[ri]], mode="markers",
                        marker=dict(color=scores, colorscale="Viridis", size=5,
                                    showscale=first, colorbar=cb,
                                    line=dict(width=0.3, color="white")),
                        text=hover,
                        hovertemplate=(f"%{{text}}<br>{dims[ci]}=%{{x:.4g}}"
                                       f"<br>{dims[ri]}=%{{y:.4g}}<extra></extra>"),
                        showlegend=False,
                    ), row=r, col=c,
                )
                first = False
            else:
                fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=r, col=c)

    # Axis configuration: bottom row gets x-labels; left col gets y-labels
    for ri in range(n):
        for ci in range(n):
            r, c = ri + 1, ci + 1
            is_lower  = ci < ri
            on_bottom = ri == n - 1
            on_left   = ci == 0
            if is_lower:
                fig.update_xaxes(
                    showticklabels=on_bottom, tickfont=dict(size=7),
                    title_text=dims[ci] if on_bottom else "",
                    title_font=dict(size=8),
                    row=r, col=c,
                )
                fig.update_yaxes(
                    showticklabels=on_left, tickfont=dict(size=7),
                    title_text=dims[ri] if on_left else "",
                    title_font=dict(size=8),
                    row=r, col=c,
                )
            else:
                fig.update_xaxes(showgrid=False, zeroline=False, showline=False,
                                 showticklabels=False, ticks="", row=r, col=c)
                fig.update_yaxes(showgrid=False, zeroline=False, showline=False,
                                 showticklabels=False, ticks="", row=r, col=c)

    # Diagonal annotations: param name labels in paper coordinates
    label_size = min(10, max(7, 80 // n))
    for i, name in enumerate(dims):
        xp = (i + 0.5) / n
        yp = 1.0 - (i + 0.5) / n
        fig.add_annotation(
            x=xp, y=yp, xref="paper", yref="paper",
            text=f"<b>{name}</b>", showarrow=False,
            font=dict(size=label_size, color="#333"),
            bgcolor="rgba(240,240,255,0.85)",
            bordercolor="#aaa", borderwidth=1, borderpad=3,
        )

    height = max(500, 120 * n)
    fig.update_layout(
        title="Hyperparameter scatter matrix",
        height=height,
        margin=dict(l=100, r=90, t=60, b=100),
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
