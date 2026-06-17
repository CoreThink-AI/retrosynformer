# RetroSynFormer 0.1.10 Release Notes

*Branch: `feature-nonuniform-dropout` — June 2026*

---

## Summary

Graceful Ctrl-C handling during training: the first interrupt finishes the
current epoch, runs route evaluation, and records the trial result to
`study.db` before stopping. A second interrupt within one second aborts
immediately.

---

## Changes

### 1. Graceful SIGINT handling (`src/retrosynformer/trainer.py`)

A custom `SIGINT` handler is installed at the start of every `train()` call
and restored (via `try/finally`) when it returns.

**First Ctrl-C:**
- Prints a message informing the user that the current epoch will finish
  before stopping.
- Sets the module-level `_interrupted` flag.
- Calls `_stop_study_callback()` if one is registered (see hypertune wiring
  below) — this invokes `study.stop()` so Optuna will not start another
  trial after the current one completes.

**Training loop response:**
- At the end of each epoch (after early-stopping check), the loop tests
  `_interrupted` and breaks if set.
- Breaking this way also forces `eval_routes_at_end = True`, so the full
  route evaluation runs before `train()` returns — the same path taken at
  the end of a normal hypertune trial.

**`train()` then returns normally**, with whatever best metrics were
achieved up to that epoch. This allows the caller to record results before
propagating the interrupt.

**Second Ctrl-C within 1 second:**
- The handler detects the sub-1-second gap, restores the original SIGINT
  handler, and raises `KeyboardInterrupt` immediately — no waiting for the
  epoch to finish.

New public API in `trainer.py`:

| Symbol | Purpose |
|--------|---------|
| `set_interrupt_callback(fn)` | Register a zero-argument callable (e.g. `study.stop`) to invoke on first Ctrl-C |
| `clear_interrupt_callback()` | Deregister the callback (called in hypertune's `finally` block) |
| `is_interrupted() → bool` | Query flag after `train()` returns to decide whether to re-raise |

### 2. Hypertune wiring (`src/retrosynformer/scripts/hypertune.py`)

`study.optimize()` is now called via an `_objective_with_interrupt` wrapper
that:

1. Calls `_trainer_mod.set_interrupt_callback(study.stop)` before each trial.
2. Calls `_trainer_mod.clear_interrupt_callback()` in a `finally` block.

After `study.optimize()` returns, `main()` checks `_trainer_mod.is_interrupted()`
and raises `KeyboardInterrupt` if set. At that point the interrupted trial's
result is already written to `study.db` (the objective returned normally),
and the best-trial summary has been printed.

### 3. Direct training wiring (`src/retrosynformer/runner.py`)

`runner.main()` likewise checks `is_interrupted()` after `trainer.train()`
returns and raises `KeyboardInterrupt`, giving `rs-train` the same clean-stop
behaviour without Optuna in the loop.
