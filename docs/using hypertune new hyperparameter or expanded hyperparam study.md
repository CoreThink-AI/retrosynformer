  Yes to all three. Optuna stores its study in a persistent database (by default SQLite), so you can:

  1. Add more trials — just call study.optimize(objective, n_trials=N) again on the same study. Optuna will continue from where it left off.
  2. Change hyperparameter ranges/values — you can widen ranges or add/remove values in the trial.suggest_* calls. Existing trials aren't affected; new trials use the updated ranges.
  3. Resume a concluded study — optuna.load_study(study_name=..., storage=...) re-opens the existing study. There's no "concluded" state in Optuna — studies only stop when you stop calling optimize().

  Let me look at how the current hypertune script handles this.

  Searched for 2 patterns (ctrl+o to expand)

retrosynformer.hypertune uses `load_if_exists=True` optuna option: rs-hypertune with the same `--study-name`  reopens the existing SQLite study. 

### Add more trials to a finished study:
`rs-hypertune -c results/config.yaml --study-name standard-v2-dropout-details --n-trials 14`

Optuna re-opens the study from results/hypertune-standard-v2-dropout-details/study.db and runs 14 additional trials using the sampler's state (prior trials inform the next ones via TPE).

### Expand or change hyperparameter ranges:
Before running `rs-hypertune` edit the `optuna` section in the YAML config, then re-run with more `--n-trials`. The new ranges take effect for all new trials; old trials in the DB are unaffected. The sampler will adapt to the wider space.

IMPORTANT: The `_validate_config` check at `hypertune.py:669` enforces `n_trials == total_combinations` when all hyperparameters are discrete. If there's a mix of continuous and discrete, Optuna treats the study as Bayesian (TPE) and any `n_trials` works.


