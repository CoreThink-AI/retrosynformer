"""SQLAlchemy ORM models for the RetroSynFormer dashboard meta-DB."""
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Study(db.Model):
    __tablename__ = "studies"

    id = db.Column(db.Integer, primary_key=True)
    study_name = db.Column(db.String(256), unique=True, nullable=False)
    db_path = db.Column(db.String(512))
    run_jsonl_path = db.Column(db.String(512))
    config_path = db.Column(db.String(512))
    objective_metric = db.Column(db.String(64))
    direction = db.Column(db.String(8), default="maximize")
    status = db.Column(db.String(16), default="unknown")  # active|complete|failed|unknown
    n_trials = db.Column(db.Integer, default=0)
    n_complete = db.Column(db.Integer, default=0)
    n_running = db.Column(db.Integer, default=0)
    n_failed = db.Column(db.Integer, default=0)
    best_score = db.Column(db.Float)
    mean_score = db.Column(db.Float)
    std_score = db.Column(db.Float)
    best_trial_number = db.Column(db.Integer)
    last_synced_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    trials = db.relationship("Trial", back_populates="study",
                             cascade="all, delete-orphan", lazy="dynamic")

    def __repr__(self):
        return f"<Study {self.study_name}>"


class Trial(db.Model):
    __tablename__ = "trials"
    __table_args__ = (db.UniqueConstraint("study_id", "trial_number"),)

    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey("studies.id"), nullable=False)
    trial_number = db.Column(db.Integer, nullable=False)
    optuna_trial_id = db.Column(db.Integer)
    state = db.Column(db.String(16), default="WAITING")  # COMPLETE|RUNNING|FAIL|WAITING
    datetime_start = db.Column(db.DateTime)
    datetime_complete = db.Column(db.DateTime)
    duration_min = db.Column(db.Float)
    params_json = db.Column(db.Text)          # {"n_heads": 4, "lr": 0.001, …}
    optuna_score = db.Column(db.Float)
    valid_loss = db.Column(db.Float)
    valid_action_accuracy = db.Column(db.Float)
    valid_route_accuracy = db.Column(db.Float)
    fraction_targets_solved = db.Column(db.Float)
    epoch_count = db.Column(db.Integer, default=0)
    trial_dir = db.Column(db.String(512))
    synced_at = db.Column(db.DateTime, default=datetime.utcnow)

    study = db.relationship("Study", back_populates="trials")

    def __repr__(self):
        return f"<Trial {self.trial_number} [{self.state}]>"
