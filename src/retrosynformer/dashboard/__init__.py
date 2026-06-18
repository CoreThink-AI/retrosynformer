"""RetroSynFormer training dashboard — Flask-Admin + REST API.

Install:  uv sync --extra dashboard
Run:      rs-dashboard --port 5050
"""
import os
from datetime import timedelta

from flask import Flask, redirect, request, session, url_for
from flask_admin import Admin

from .extensions import limiter
from .models import EpochRecord, Study, Trial, TrialHyperparams, db
from .views import (DashboardIndexView, EpochRecordAdmin, StudyAdmin,
                    TrialAdmin, TrialHyperparamsAdmin)
from .views import bp as dashboard_bp

_INSECURE_DEFAULT_KEY = "dev-key-change-in-prod"


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

    secret_key = os.environ.get("SECRET_KEY", _INSECURE_DEFAULT_KEY)

    app.config.update(
        SECRET_KEY=secret_key,
        SQLALCHEMY_DATABASE_URI=db_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        RESULTS_ROOT=results_root,
        CLOUD_RUN_URL=cloud_run_url,
        DEBUG=debug,
        # --- Session cookie hardening ---
        SESSION_COOKIE_SECURE=True,      # only sent over HTTPS
        SESSION_COOKIE_HTTPONLY=True,    # inaccessible to JavaScript
        SESSION_COOKIE_SAMESITE="Lax",  # blocks cross-site POST forgery
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        SESSION_REFRESH_EACH_REQUEST=True,
        # --- Rate-limiter defaults (overridden per-route as needed) ---
        RATELIMIT_DEFAULT="200 per day;50 per hour",
        RATELIMIT_HEADERS_ENABLED=True,
    )

    # --- Auth setup ---------------------------------------------------------
    _pw = os.environ.get("DASHBOARD_PASSWORD", "")
    if _pw:
        from werkzeug.security import generate_password_hash
        if secret_key == _INSECURE_DEFAULT_KEY:
            raise ValueError(
                "FATAL: Set a strong SECRET_KEY in .env.dashboard before enabling "
                "authentication. Generate one with:\n"
                "  python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        app.config["DASHBOARD_USERNAME"] = os.environ.get("DASHBOARD_USERNAME", "admin")
        app.config["DASHBOARD_PASSWORD_HASH"] = generate_password_hash(_pw)
        app.config["AUTH_REQUIRED"] = True
    else:
        app.config["AUTH_REQUIRED"] = False

    @app.before_request
    def _require_login():
        if not app.config.get("AUTH_REQUIRED"):
            return
        if request.path in ("/login", "/logout"):
            return
        if request.path.startswith(("/static/", "/admin/static/")):
            return
        if not session.get("logged_in"):
            return redirect(url_for("dashboard.login", next=request.url))

    @app.after_request
    def _security_headers(response):
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # HSTS: tell browsers to always use HTTPS for this host for 1 year
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # CSP: allow Bootstrap/Plotly CDN; 'unsafe-inline' needed for Plotly
        # chart initialisation blocks in dashboard templates.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self' https://cdn.jsdelivr.net"
        )
        return response
    # ------------------------------------------------------------------------

    limiter.init_app(app)
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
    admin.add_view(TrialHyperparamsAdmin(
        TrialHyperparams, db.session, name="Hyperparams", endpoint="hyperparams_admin",
    ))
    admin.add_view(EpochRecordAdmin(
        EpochRecord, db.session, name="Epoch Records", endpoint="epoch_admin",
    ))

    app.register_blueprint(dashboard_bp)

    if initial_sync:
        from .sync import sync_all
        with app.app_context():
            result = sync_all(results_root)
            app.logger.info(f"Initial sync: {result}")

    return app
