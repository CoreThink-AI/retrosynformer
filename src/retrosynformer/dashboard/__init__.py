"""RetroSynFormer training dashboard — Flask-Admin + REST API.

Install:  uv sync --extra dashboard
Run:      rs-dashboard --port 5050
"""
import os

from flask import Flask
from flask_admin import Admin

from .models import Study, Trial, db
from .views import DashboardIndexView, StudyAdmin, TrialAdmin
from .views import bp as dashboard_bp


def create_app(
    results_root: str | None = None,
    db_url: str | None = None,
    initial_sync: bool = True,
    cloud_run_url: str | None = None,
    debug: bool = False,
) -> Flask:
    results_root = os.path.abspath(results_root or os.environ.get("RESULTS_ROOT", "results/"))
    db_url = db_url or os.environ.get("DASHBOARD_DB_URL",
                                      f"sqlite:///{results_root}/dashboard.db")
    cloud_run_url = cloud_run_url or os.environ.get("CLOUD_RUN_URL", "")

    os.makedirs(results_root, exist_ok=True)
    app = Flask(__name__, template_folder="templates")
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-key-change-in-prod"),
        SQLALCHEMY_DATABASE_URI=db_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        RESULTS_ROOT=results_root,
        CLOUD_RUN_URL=cloud_run_url,
        DEBUG=debug,
    )

    db.init_app(app)

    with app.app_context():
        db.create_all()

    admin = Admin(
        app,
        name="RetroSynFormer",
        index_view=DashboardIndexView(name="Home", url="/admin"),
    )
    admin.add_view(StudyAdmin(Study, db.session, name="Studies", endpoint="study_admin"))
    admin.add_view(TrialAdmin(Trial, db.session, name="Trials", endpoint="trial_admin"))

    app.register_blueprint(dashboard_bp)

    if initial_sync:
        from .sync import sync_all
        with app.app_context():
            result = sync_all(results_root)
            app.logger.info(f"Initial sync: {result}")

    return app
