"""Flask-Admin ModelViews and custom Blueprint for the dashboard."""
import json
import os

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for
from flask_admin import Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView

from .models import Study, Trial, db
from .sync import sync_all, sync_study


# ---------------------------------------------------------------------------
# Flask-Admin ModelViews
# ---------------------------------------------------------------------------

class StudyAdmin(ModelView):
    column_list = [
        "study_name", "status", "n_trials", "n_complete", "n_running",
        "best_score", "best_trial_number", "objective_metric", "last_synced_at",
    ]
    column_searchable_list = ["study_name"]
    column_filters = ["status", "best_score", "n_trials"]
    column_default_sort = ("last_synced_at", True)
    can_create = False
    can_delete = True
    can_edit = False
    column_formatters = {
        "best_score": lambda v, c, m, n: f"{m.best_score:.4f}" if m.best_score is not None else "—",
        "last_synced_at": lambda v, c, m, n: m.last_synced_at.strftime("%H:%M:%S") if m.last_synced_at else "—",
    }


class TrialAdmin(ModelView):
    column_list = [
        "study", "trial_number", "state", "epoch_count",
        "optuna_score", "valid_loss", "valid_action_accuracy",
        "valid_route_accuracy", "fraction_targets_solved", "duration_min",
    ]
    column_searchable_list = []
    column_filters = ["state", "optuna_score", "fraction_targets_solved", "epoch_count"]
    column_default_sort = ("synced_at", True)
    can_create = False
    can_delete = False
    can_edit = False
    column_formatters = {
        "optuna_score": lambda v, c, m, n: f"{m.optuna_score:.4f}" if m.optuna_score is not None else "—",
        "valid_loss": lambda v, c, m, n: f"{m.valid_loss:.4f}" if m.valid_loss is not None else "—",
        "valid_action_accuracy": lambda v, c, m, n: f"{m.valid_action_accuracy:.4f}" if m.valid_action_accuracy is not None else "—",
        "valid_route_accuracy": lambda v, c, m, n: f"{m.valid_route_accuracy:.4f}" if m.valid_route_accuracy is not None else "—",
        "fraction_targets_solved": lambda v, c, m, n: f"{m.fraction_targets_solved:.4f}" if m.fraction_targets_solved is not None else "—",
    }

    def _curves_link(self, context, model, name):
        if model.trial_dir is None:
            return "—"
        from markupsafe import Markup
        study_name = model.study.study_name if model.study else "?"
        url = url_for("dashboard.trial_curves", study_name=study_name,
                      trial_num=model.trial_number)
        return Markup(f'<a href="{url}">📈 curves</a>')

    column_extra_row_actions = []
    column_list = [
        "study", "trial_number", "state", "epoch_count",
        "optuna_score", "valid_loss", "valid_action_accuracy",
        "valid_route_accuracy", "fraction_targets_solved", "duration_min",
    ]


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


@bp.get("/")
def index():
    studies = Study.query.order_by(Study.last_synced_at.desc()).all()
    active = [s for s in studies if s.status == "active"]
    cloud_run_url = current_app.config.get("CLOUD_RUN_URL", "")
    return render_template(
        "dashboard/index.html",
        studies=studies,
        active=active,
        cloud_run_url=cloud_run_url,
    )


@bp.get("/trial/<study_name>/<int:trial_num>/curves")
def trial_curves(study_name: str, trial_num: int):
    study = Study.query.filter_by(study_name=study_name).first_or_404()
    trial = Trial.query.filter_by(study_id=study.id, trial_number=trial_num).first_or_404()
    epochs_data = []
    if trial.trial_dir:
        jsonl_path = os.path.join(trial.trial_dir, "train_progress.jsonl")
        if os.path.exists(jsonl_path):
            import json as _json
            rows = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(_json.loads(line))
                        except _json.JSONDecodeError:
                            pass
            # Reset detection: keep only the last contiguous run
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

    return render_template(
        "dashboard/trial_curves.html",
        study=study,
        trial=trial,
        params=params,
        epochs_data=json.dumps(epochs_data),
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
