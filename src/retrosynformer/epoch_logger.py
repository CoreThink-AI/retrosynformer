"""EpochLogger — configurable per-epoch metric accumulator and JSONL writer.

Any code in the training pipeline, including deeply nested functions, can
call :func:`EpochLogger.update` to register a metric for the current epoch.
At the end of each epoch the trainer calls :func:`EpochLogger.flush` to
write the configured subset of fields to disk.

Configuration (``logging:`` section of ``model.config.yaml``)::

    logging:
      # List the field names you want written to train_progress.jsonl.
      # Omit this key (or omit the whole logging: section) to write every
      # accumulated field — i.e. backward-compatible default.
      fields:
        - epoch
        - train_loss
        - valid_loss
        - learning_rate
        - gradient_norm
        - elapsed_seconds
        - timestamp
        - is_best
        - epochs_without_improvement
        - n_lr_reductions
        - best_valid_route_accuracy
        - study_name
        - trial_number
        - config_hash

Extending with custom metrics
-----------------------------
Call :func:`EpochLogger.update` from anywhere — including inside
``train_one_epoch``, a custom loss function, or a callback::

    from retrosynformer.epoch_logger import EpochLogger

    # Inside any training code:
    EpochLogger.update("my_custom_metric", value)

Register computed fields (evaluated fresh at each flush)::

    EpochLogger.register_provider(
        "throughput_samples_per_sec",
        lambda: n_samples / EpochLogger.get("seconds_per_epoch", 1),
    )

Values set via :func:`set_persistent` are included in every epoch record
unchanged — useful for ``study_name``, ``trial_number``, ``config_hash``.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable


class EpochLogger:
    """Module-level singleton for training-state accumulation and JSONL logging.

    All methods are classmethods; no instantiation needed.  One instance of
    state exists per Python process, so this is safe for single-threaded
    training but not for multi-threaded use of :class:`RetroTrainer`.
    """

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    # Values that change every epoch (reset by begin_epoch)
    _state: dict[str, Any] = {}

    # Values that are constant for the whole training run (repeated every line)
    _persistent: dict[str, Any] = {}

    # Zero-argument callables evaluated fresh at each flush()
    _providers: dict[str, Callable[[], Any]] = {}

    # Config
    _fields: list[str] | None = None   # None → write all
    _jsonl_path: str | None = None
    _train_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @classmethod
    def configure(
        cls,
        logging_config: dict,
        jsonl_path: str | None = None,
        train_start_time: float | None = None,
    ) -> None:
        """Initialise the logger for a new training run.

        Call once before the epoch loop starts.

        Parameters
        ----------
        logging_config:
            The ``logging:`` section from ``model.config.yaml``.
            Pass ``{}`` (or ``config.get("logging", {})``) when the section
            is absent — this writes every accumulated field.
        jsonl_path:
            Path to the JSONL file.  ``flush()`` appends one line per call.
            If ``None``, records are returned but not written.
        train_start_time:
            ``time.time()`` at the start of the training loop.  Used by the
            built-in ``elapsed_seconds`` provider.  Defaults to now.
        """
        cls._fields = logging_config.get("fields", None)
        cls._jsonl_path = str(jsonl_path) if jsonl_path else None
        cls._train_start_time = train_start_time if train_start_time is not None else time.time()
        cls._state = {}
        cls._persistent = {}
        cls._providers = {}

        # Built-in computed providers — always registered; only written if
        # their name is in _fields (or _fields is None).
        cls.register_provider(
            "elapsed_seconds",
            lambda: time.time() - cls._train_start_time,
        )
        cls.register_provider(
            "timestamp",
            lambda: __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        )

    @classmethod
    def set_persistent(cls, **kwargs: Any) -> None:
        """Register values written unchanged in every epoch record.

        Call after :func:`configure`.  Typical values: ``study_name``,
        ``trial_number``, ``config_hash``, ``n_epochs``.
        """
        cls._persistent.update(kwargs)

    @classmethod
    def register_provider(cls, name: str, fn: Callable[[], Any]) -> None:
        """Register a zero-argument callable evaluated at :func:`flush` time.

        Providers are evaluated after :func:`update_many` so they can read
        accumulated state via :func:`get`.  If the callable raises, the field
        is written as ``null``.

        Example::

            EpochLogger.register_provider(
                "throughput",
                lambda: n_samples / EpochLogger.get("seconds_per_epoch", 1),
            )
        """
        cls._providers[name] = fn

    # ------------------------------------------------------------------
    # Per-epoch API
    # ------------------------------------------------------------------

    @classmethod
    def begin_epoch(cls, epoch: int) -> None:
        """Reset per-epoch state and set ``epoch``.  Call at epoch start."""
        cls._state = {"epoch": epoch}

    @classmethod
    def update(cls, key: str, value: Any) -> None:
        """Set a single metric for the current epoch.

        Safe to call from inside ``train_one_epoch``, callbacks, or any
        nested training function — the value will appear in the JSONL record
        for that epoch.
        """
        cls._state[key] = value

    @classmethod
    def update_many(cls, **kwargs: Any) -> None:
        """Set multiple metrics for the current epoch."""
        cls._state.update(kwargs)

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Read a value from the current epoch state or persistent store."""
        return cls._state.get(key, cls._persistent.get(key, default))

    @classmethod
    def flush(cls) -> dict:
        """Assemble and write the epoch record to the JSONL file.

        Field precedence (later wins): persistent → per-epoch state →
        computed providers.

        Returns the record dict regardless of whether it was written to disk.
        """
        full: dict[str, Any] = {}
        full.update(cls._persistent)
        full.update(cls._state)

        # Evaluate providers (computed fields, e.g. elapsed_seconds, timestamp)
        for name, fn in cls._providers.items():
            try:
                full[name] = fn()
            except Exception:
                full[name] = None

        # Apply field filter from config
        if cls._fields is not None:
            record = {k: full[k] for k in cls._fields if k in full}
        else:
            record = full

        if cls._jsonl_path:
            with open(cls._jsonl_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

        return record
