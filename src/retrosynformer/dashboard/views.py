"""Flask-Admin ModelViews and custom Blueprint for the dashboard."""
import hmac
import json
import os
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

from flask import (Blueprint, current_app, jsonify, redirect, render_template,
                   request, session, url_for)
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from markupsafe import Markup
from sqlalchemy import func
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import limiter
from .models import EpochRecord, Study, Trial, TrialHyperparams, db
from .sync import sync_all, sync_study

# Used as the comparison target when the submitted username doesn't match,
# so check_password_hash always runs and response time is constant.
_DUMMY_HASH = generate_password_hash("dummy-timing-safety")

STALE_TRIAL_HOURS = 2  # RUNNING trials not synced within this window are shown as stopped


def _is_stale(trial) -> bool:
    return (
        trial.state == "RUNNING"
        and trial.synced_at is not None
        and (datetime.utcnow() - trial.synced_at) > timedelta(hours=STALE_TRIAL_HOURS)
    )


# ---------------------------------------------------------------------------
# Flask-Admin ModelViews
# ---------------------------------------------------------------------------

class StudyAdmin(ModelView):
    column_list = [
        "study_name", "status", "started_at", "completed_at",
        "n_trials", "n_complete", "n_running",
        "best_score", "best_trial_number", "objective_metric", "last_synced_at",
    ]
    column_sortable_list = [
        "study_name", "status", "started_at", "completed_at",
        "n_trials", "n_complete", "n_running",
        "best_score", "best_trial_number", "objective_metric", "last_synced_at",
    ]
    column_searchable_list = ["study_name"]
    column_filters = ["status", "best_score", "n_trials"]
    column_default_sort = ("started_at", True)
    can_create = False
    can_delete = True
    can_edit = False
    can_view_details = True
    column_formatters = {
        "best_score":    lambda v, c, m, n: f"{m.best_score:.4f}" if m.best_score is not None else "—",
        "started_at":    lambda v, c, m, n: m.started_at.strftime("%Y-%m-%d %H:%M")    if m.started_at    else "—",
        "completed_at":  lambda v, c, m, n: m.completed_at.strftime("%Y-%m-%d %H:%M")  if m.completed_at  else "—",
        "last_synced_at": lambda v, c, m, n: m.last_synced_at.strftime("%H:%M:%S")     if m.last_synced_at else "—",
    }


class TrialAdmin(ModelView):
    column_list = [
        "study", "trial_number", "state",
        "datetime_start", "datetime_complete", "duration_min",
        "epoch_count",
        "optuna_score", "valid_loss", "valid_action_accuracy",
        "valid_route_accuracy", "fraction_targets_solved",
        "synced_at", "curves",
    ]
    column_sortable_list = [
        ("study", "study.study_name"), "trial_number", "state",
        "datetime_start", "datetime_complete", "duration_min",
        "epoch_count",
        "optuna_score", "valid_loss", "valid_action_accuracy",
        "valid_route_accuracy", "fraction_targets_solved",
        "synced_at",
    ]
    column_searchable_list = ["study.study_name"]
    column_filters = [
        "state", "study.study_name",
        "datetime_start", "datetime_complete", "duration_min",
        "optuna_score", "valid_loss", "valid_action_accuracy",
        "valid_route_accuracy", "fraction_targets_solved",
        "epoch_count",
    ]
    column_default_sort = ("datetime_start", True)
    can_create = False
    can_delete = False
    can_edit = False
    column_formatters = {
        "study": lambda v, c, m, n: Markup(
            f'<a href="{url_for("study_admin.details_view", id=m.study.id)}">'
            f'{m.study.study_name}</a>'
        ) if m.study else "?",
        "state": lambda v, c, m, n: Markup(
            f'<span class="badge badge-stopped">stopped</span>'
            if _is_stale(m) else
            f'<span class="badge badge-{m.state.lower()}">{m.state}</span>'
        ),
        "datetime_start":    lambda v, c, m, n: m.datetime_start.strftime("%Y-%m-%d %H:%M")    if m.datetime_start    else "—",
        "datetime_complete": lambda v, c, m, n: m.datetime_complete.strftime("%Y-%m-%d %H:%M") if m.datetime_complete else "—",
        "synced_at":         lambda v, c, m, n: m.synced_at.strftime("%Y-%m-%d %H:%M")         if m.synced_at         else "—",
        "optuna_score":              lambda v, c, m, n: f"{m.optuna_score:.4f}"              if m.optuna_score              is not None else "—",
        "valid_loss":                lambda v, c, m, n: f"{m.valid_loss:.4f}"                if m.valid_loss                is not None else "—",
        "valid_action_accuracy":     lambda v, c, m, n: f"{m.valid_action_accuracy:.4f}"     if m.valid_action_accuracy     is not None else "—",
        "valid_route_accuracy":      lambda v, c, m, n: f"{m.valid_route_accuracy:.4f}"      if m.valid_route_accuracy      is not None else "—",
        "fraction_targets_solved":   lambda v, c, m, n: f"{m.fraction_targets_solved:.4f}"   if m.fraction_targets_solved   is not None else "—",
        "duration_min":              lambda v, c, m, n: f"{m.duration_min:.1f}"              if m.duration_min              is not None else "—",
        "curves": lambda v, c, m, n: Markup(
            f'<a href="{url_for("dashboard.trial_curves", study_name=m.study.study_name, trial_num=m.trial_number)}">curves</a>'
        ) if m.trial_dir and m.study else "—",
    }


class EpochRecordAdmin(ModelView):
    """Per-epoch learning curve rows from train_progress.jsonl."""
    column_list = [
        "trial_id", "epoch", "train_loss", "train_action_accuracy",
        "valid_loss", "valid_action_accuracy", "valid_route_accuracy",
        "seconds_per_epoch", "lr",
    ]
    column_sortable_list = [
        "trial_id", "epoch", "train_loss", "valid_loss",
        "valid_action_accuracy", "valid_route_accuracy",
    ]
    column_filters = ["trial_id", "epoch", "valid_action_accuracy"]
    column_default_sort = [("trial_id", False), ("epoch", False)]
    can_create = False
    can_delete = False
    can_edit = False
    column_formatters = {
        k: (lambda v, c, m, n, _k=k: f"{getattr(m, _k):.4f}" if getattr(m, _k) is not None else "—")
        for k in ("train_loss", "train_action_accuracy", "valid_loss",
                  "valid_action_accuracy", "valid_route_accuracy", "lr")
    }


class TrialHyperparamsAdmin(ModelView):
    """Flat hyperparameter table with completeness flags (mirrors all_hyperparams.csv)."""
    column_list = [
        "trial_id",
        "incomplete_reason", "is_incomplete", "is_early_stopped", "is_jsonl_unreliable",
        "cfg_dataset", "cfg_n_heads", "cfg_n_layers", "cfg_head_dim",
        "cfg_attn_pdrop", "cfg_embd_pdrop", "cfg_resid_pdrop",
        "cfg_lr", "cfg_n_epochs",
        "jsonl_last_epoch", "total_jsonl_epochs", "epoch_ran_fraction",
        "max_complete_epoch_in_study",
        "estimated_valid_action_accuracy", "estimated_valid_route_accuracy",
        "git_hash_short", "git_message",
        "synced_at",
    ]
    column_sortable_list = [
        "trial_id", "incomplete_reason", "cfg_dataset",
        "cfg_n_heads", "cfg_n_layers", "cfg_head_dim",
        "cfg_attn_pdrop", "cfg_lr", "cfg_n_epochs",
        "jsonl_last_epoch", "epoch_ran_fraction",
        "estimated_valid_action_accuracy",
    ]
    column_filters = [
        "incomplete_reason", "is_incomplete", "is_jsonl_unreliable",
        "cfg_dataset", "cfg_n_heads", "cfg_n_layers",
    ]
    column_default_sort = ("trial_id", False)
    can_create = False
    can_delete = False
    can_edit = False
    can_view_details = True
    column_formatters = {
        k: (lambda v, c, m, n, _k=k: f"{getattr(m, _k):.4f}" if getattr(m, _k) is not None else "—")
        for k in ("cfg_lr", "cfg_attn_pdrop", "cfg_embd_pdrop", "cfg_resid_pdrop",
                  "epoch_ran_fraction",
                  "estimated_valid_action_accuracy", "estimated_valid_route_accuracy")
    }


# ---------------------------------------------------------------------------
# Custom Dashboard Index
# ---------------------------------------------------------------------------

class DashboardIndexView(AdminIndexView):
    @expose("/")
    def index(self):
        return redirect(url_for("dashboard.index"))


# ---------------------------------------------------------------------------
# Blueprint: custom pages + REST endpoints
# ---------------------------------------------------------------------------

bp = Blueprint("dashboard", __name__, template_folder="templates")


@bp.app_context_processor
def _inject_studies_for_nav():
    """Make all studies available in every template for the Plots dropdown."""
    try:
        studies = Study.query.order_by(Study.last_synced_at.desc()).all()
    except Exception:
        studies = []
    return {"nav_studies": studies}


@bp.get("/")
def index():
    studies = Study.query.order_by(Study.last_synced_at.desc()).all()
    active = [s for s in studies if s.status == "active"]
    cloud_run_url = current_app.config.get("CLOUD_RUN_URL", "")
    study_epochs = dict(
        db.session.query(Trial.study_id, func.max(Trial.epoch_count))
        .group_by(Trial.study_id)
        .all()
    )
    return render_template(
        "dashboard/index.html",
        studies=studies,
        active=active,
        cloud_run_url=cloud_run_url,
        now=datetime.utcnow(),
        stale_hours=STALE_TRIAL_HOURS,
        study_epochs=study_epochs,
    )


@bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"], error_message="Too many login attempts — try again in a minute.")
@limiter.limit("20 per hour", methods=["POST"], error_message="Too many login attempts — try again later.")
def login():
    if session.get("logged_in"):
        return redirect(url_for("dashboard.index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        expected = current_app.config.get("DASHBOARD_USERNAME", "admin")
        pw_hash = current_app.config.get("DASHBOARD_PASSWORD_HASH", _DUMMY_HASH)
        # Always run check_password_hash (constant time) and compare username
        # with hmac.compare_digest to prevent timing-based username enumeration.
        username_ok = hmac.compare_digest(username, expected)
        password_ok = check_password_hash(pw_hash, password)
        if username_ok and password_ok and current_app.config.get("AUTH_REQUIRED"):
            session.permanent = True
            session["logged_in"] = True
            next_url = request.args.get("next", "")
            if next_url and _is_safe_url(next_url):
                return redirect(next_url)
            return redirect(url_for("dashboard.index"))
        error = "Invalid credentials"
    return render_template("dashboard/login.html", error=error)


@bp.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard.login"))


def _is_safe_url(target: str) -> bool:
    ref = urlparse(request.host_url)
    test = urlparse(urljoin(request.host_url, target))
    return (
        test.scheme in ("http", "https")
        and ref.netloc == test.netloc
        and not test.fragment  # block #-based open-redirect bypasses
    )


@bp.get("/study/<study_name>/parcoords")
def study_parcoords(study_name: str):
    from .plots import (build_parcoords_figure, build_optimization_history_figure,
                        build_param_importances_figure, load_optuna_study)

    study = Study.query.filter_by(study_name=study_name).first_or_404()
    if not study.db_path or not os.path.exists(study.db_path):
        return "study.db not found", 404

    try:
        optuna_study = load_optuna_study(study.db_path, study.study_name)
    except Exception as exc:
        return f"Could not load Optuna study: {exc}", 500

    parcoords_fig = build_parcoords_figure(optuna_study)
    history_fig = build_optimization_history_figure(optuna_study)
    importance_fig = build_param_importances_figure(optuna_study)

    n_complete = sum(1 for t in optuna_study.trials if t.state.name == "COMPLETE")

    return render_template(
        "dashboard/parcoords.html",
        study=study,
        n_complete=n_complete,
        parcoords_json=parcoords_fig.to_json(),
        history_json=history_fig.to_json(),
        importance_json=importance_fig.to_json() if importance_fig else None,
    )


@bp.get("/trial/<study_name>/<int:trial_num>/curves")
def trial_curves(study_name: str, trial_num: int):
    from .plots import build_trial_figure

    study = Study.query.filter_by(study_name=study_name).first_or_404()
    trial = Trial.query.filter_by(study_id=study.id, trial_number=trial_num).first_or_404()
    epochs_data = []
    if trial.trial_dir:
        jsonl_path = os.path.join(trial.trial_dir, "train_progress.jsonl")
        if os.path.exists(jsonl_path):
            rows = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if rows:
                last_reset = 0
                for i in range(1, len(rows)):
                    if rows[i].get("epoch", i) <= rows[i - 1].get("epoch", i - 1):
                        last_reset = i
                epochs_data = rows[last_reset:]

    params = {}
    if trial.params_json:
        try:
            params = json.loads(trial.params_json)
        except json.JSONDecodeError:
            pass

    title = f"{study.study_name} / trial_{trial.trial_number:03d}"
    fig = build_trial_figure(epochs_data, title=title)

    return render_template(
        "dashboard/trial_curves.html",
        study=study,
        trial=trial,
        params=params,
        fig_json=fig.to_json(),
    )


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------

@bp.post("/api/v1/sync")
def api_sync():
    root = current_app.config["RESULTS_ROOT"]
    result = sync_all(root)
    return jsonify({"ok": True, "data": result})


@bp.post("/api/v1/studies/<study_name>/sync")
def api_sync_study(study_name: str):
    study = Study.query.filter_by(study_name=study_name).first_or_404()
    updated = sync_study(study.db_path, current_app.config["RESULTS_ROOT"], force=True)
    return jsonify({"ok": True, "data": {"updated": updated}})


@bp.get("/api/v1/studies")
def api_list_studies():
    q = Study.query
    if status := request.args.get("status"):
        q = q.filter(Study.status == status)
    if (min_score := request.args.get("min_score", type=float)) is not None:
        q = q.filter(Study.best_score >= min_score)
    if study_filter := request.args.get("study"):
        q = q.filter(Study.study_name.contains(study_filter))
    rows = q.order_by(Study.last_synced_at.desc()).all()
    data = [_study_to_dict(s) for s in rows]
    return jsonify({"ok": True, "data": data})


@bp.get("/api/v1/studies/<study_name>")
def api_get_study(study_name: str):
    study = Study.query.filter_by(study_name=study_name).first_or_404()
    return jsonify({"ok": True, "data": _study_to_dict(study)})


@bp.get("/api/v1/studies/<study_name>/trials")
def api_list_trials(study_name: str):
    study = Study.query.filter_by(study_name=study_name).first_or_404()
    q = Trial.query.filter_by(study_id=study.id)
    if state := request.args.get("state"):
        q = q.filter(Trial.state == state.upper())
    if (min_score := request.args.get("min_score", type=float)) is not None:
        q = q.filter(Trial.optuna_score >= min_score)
    trials = q.order_by(Trial.trial_number).all()
    return jsonify({"ok": True, "data": [_trial_to_dict(t) for t in trials]})


@bp.get("/api/v1/studies/<study_name>/trials/<int:trial_num>")
def api_get_trial(study_name: str, trial_num: int):
    study = Study.query.filter_by(study_name=study_name).first_or_404()
    trial = Trial.query.filter_by(study_id=study.id, trial_number=trial_num).first_or_404()
    return jsonify({"ok": True, "data": _trial_to_dict(trial)})


@bp.get("/api/v1/studies/<study_name>/trials/<int:trial_num>/epochs")
def api_trial_epochs(study_name: str, trial_num: int):
    study = Study.query.filter_by(study_name=study_name).first_or_404()
    trial = Trial.query.filter_by(study_id=study.id, trial_number=trial_num).first_or_404()
    if not trial.trial_dir:
        return jsonify({"ok": False, "data": None, "error": "trial_dir not found"}), 404
    jsonl_path = os.path.join(trial.trial_dir, "train_progress.jsonl")
    if not os.path.exists(jsonl_path):
        return jsonify({"ok": True, "data": []})
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return jsonify({"ok": True, "data": rows})


@bp.post("/api/v1/sync-remote")
def api_sync_remote():
    body = request.get_json(silent=True) or {}
    host = body.get("host", "taco")
    remote_path = body.get("remote_path", "code/corethink/retrosynformer/results/")
    dry_run = body.get("dry_run", False)
    from retrosynformer.rsync import sync, DEFAULT_INCLUDES
    rc = sync(
        src=f"{host}:{remote_path}",
        dst=current_app.config["RESULTS_ROOT"],
        includes=list(DEFAULT_INCLUDES) + ["pred_routes_train_progress.json", "model.config.yaml"],
        verbose=False,
        dry_run=dry_run,
    )
    return jsonify({"ok": rc == 0, "data": {"exit_code": rc}})


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _study_to_dict(s: Study) -> dict:
    return {
        "study_name": s.study_name,
        "status": s.status,
        "objective_metric": s.objective_metric,
        "direction": s.direction,
        "n_trials": s.n_trials,
        "n_complete": s.n_complete,
        "n_running": s.n_running,
        "n_failed": s.n_failed,
        "best_score": s.best_score,
        "mean_score": s.mean_score,
        "std_score": s.std_score,
        "best_trial_number": s.best_trial_number,
        "last_synced_at": s.last_synced_at.isoformat() if s.last_synced_at else None,
        "db_path": s.db_path,
        "config_path": s.config_path,
    }


def _trial_to_dict(t: Trial) -> dict:
    params = {}
    if t.params_json:
        try:
            params = json.loads(t.params_json)
        except json.JSONDecodeError:
            pass
    return {
        "trial_number": t.trial_number,
        "state": t.state,
        "epoch_count": t.epoch_count,
        "duration_min": t.duration_min,
        "optuna_score": t.optuna_score,
        "valid_loss": t.valid_loss,
        "valid_action_accuracy": t.valid_action_accuracy,
        "valid_route_accuracy": t.valid_route_accuracy,
        "fraction_targets_solved": t.fraction_targets_solved,
        "params": params,
        "trial_dir": t.trial_dir,
        "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
        "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
        "synced_at": t.synced_at.isoformat() if t.synced_at else None,
    }
