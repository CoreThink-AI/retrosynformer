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
    started_at = db.Column(db.DateTime)    # earliest trial datetime_start
    completed_at = db.Column(db.DateTime)  # latest trial datetime_complete (or latest start)
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
    epoch_records = db.relationship("EpochRecord", back_populates="trial",
                                    cascade="all, delete-orphan", lazy="dynamic")
    hyperparams = db.relationship("TrialHyperparams", back_populates="trial",
                                  uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Trial {self.trial_number} [{self.state}]>"


class EpochRecord(db.Model):
    """Per-epoch training metrics from train_progress.jsonl (last contiguous run only)."""
    __tablename__ = "epoch_records"
    __table_args__ = (db.UniqueConstraint("trial_id", "epoch"),)

    id = db.Column(db.Integer, primary_key=True)
    trial_id = db.Column(db.Integer, db.ForeignKey("trials.id"), nullable=False)
    epoch = db.Column(db.Integer, nullable=False)           # 0-indexed epoch number
    train_loss = db.Column(db.Float)
    train_action_accuracy = db.Column(db.Float)
    valid_loss = db.Column(db.Float)
    valid_action_accuracy = db.Column(db.Float)
    valid_route_accuracy = db.Column(db.Float)
    seconds_per_epoch = db.Column(db.Float)
    lr = db.Column(db.Float)

    trial = db.relationship("Trial", back_populates="epoch_records")

    def __repr__(self):
        return f"<EpochRecord trial={self.trial_id} epoch={self.epoch}>"


class TrialHyperparams(db.Model):
    """Flat hyperparameter row for each trial — mirrors all_hyperparams.csv.

    Populated by sync from model.config.yaml + train_progress.jsonl.
    Completeness columns (is_incomplete, etc.) are recomputed across all
    trials in the study after each sync pass.
    """
    __tablename__ = "trial_hyperparams"

    id = db.Column(db.Integer, primary_key=True)
    trial_id = db.Column(db.Integer, db.ForeignKey("trials.id"),
                         unique=True, nullable=False)

    # --- Model architecture ---
    cfg_n_heads = db.Column(db.Integer)
    cfg_n_layers = db.Column(db.Integer)
    cfg_head_dim = db.Column(db.Integer)
    cfg_hidden_size = db.Column(db.Integer)
    cfg_max_ep_len = db.Column(db.Integer)
    cfg_activation_function = db.Column(db.String(32))
    cfg_action_tanh = db.Column(db.Boolean)

    # --- Dropout ---
    cfg_attn_pdrop = db.Column(db.Float)
    cfg_embd_pdrop = db.Column(db.Float)
    cfg_resid_pdrop = db.Column(db.Float)
    cfg_use_structured_dropout = db.Column(db.Boolean)
    cfg_structured_dropout_bottleneck = db.Column(db.Float)
    cfg_structured_dropout_rate = db.Column(db.Float)

    # --- Optimizer ---
    cfg_lr = db.Column(db.Float)
    cfg_momentum = db.Column(db.Float)

    # --- Training schedule ---
    cfg_batch_size = db.Column(db.Integer)
    cfg_n_epochs = db.Column(db.Integer)
    cfg_early_stopping_patience = db.Column(db.Integer)
    cfg_lr_scheduler_patience = db.Column(db.Integer)
    cfg_loss = db.Column(db.String(64))

    # --- Dataset ---
    cfg_action_dim = db.Column(db.Integer)
    cfg_dataset = db.Column(db.String(32))          # "small" | "standard" | "large"
    cfg_fp_dim = db.Column(db.Integer)
    cfg_n_in_state = db.Column(db.Integer)
    cfg_valid_set = db.Column(db.String(32))
    cfg_random_state = db.Column(db.Integer)

    # --- Reward ---
    cfg_bb_reward = db.Column(db.Float)
    cfg_dead_end_reward = db.Column(db.Float)
    cfg_intermediate_reward = db.Column(db.Float)

    # --- Evaluation ---
    cfg_beam_width = db.Column(db.Integer)
    cfg_eval_routes_frequency = db.Column(db.Integer)

    # --- Epoch counters (from JSONL) ---
    jsonl_last_epoch = db.Column(db.Integer, default=0)     # 0-based last epoch in last run
    total_jsonl_epochs = db.Column(db.Integer)              # raw line count (>last_epoch if restarted)
    epoch_ran_fraction = db.Column(db.Float)                # epoch_count / cfg_n_epochs

    # --- Completeness analysis (computed across all trials in the study) ---
    max_complete_epoch_in_study = db.Column(db.Float)
    is_incomplete = db.Column(db.Boolean, default=False)
    is_early_stopped = db.Column(db.Boolean, default=False)
    is_jsonl_unreliable = db.Column(db.Boolean, default=False)
    # "complete"|"early_stopped"|"complete_restarted"|"killed"|"running"|
    # "nearly_complete"|"no_jsonl"|"no_complete_ref"
    incomplete_reason = db.Column(db.String(32))

    # --- Estimated metrics for incomplete/running trials ---
    estimated_valid_action_accuracy = db.Column(db.Float)
    estimated_valid_route_accuracy = db.Column(db.Float)
    estimation_n_ref = db.Column(db.Integer)

    # --- Git provenance ---
    git_hash = db.Column(db.String(40))
    git_hash_short = db.Column(db.String(8))
    git_message = db.Column(db.String(256))

    # --- Optuna searched parameters (what the sampler chose) ---
    optuna_searched = db.Column(db.String(256))   # pipe-separated param names

    synced_at = db.Column(db.DateTime, default=datetime.utcnow)

    trial = db.relationship("Trial", back_populates="hyperparams")

    def __repr__(self):
        return f"<TrialHyperparams trial={self.trial_id}>"
