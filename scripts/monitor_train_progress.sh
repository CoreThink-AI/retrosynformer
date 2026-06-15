tail  -q -n1 results/hypertune/trial_???/train_progress.jsonl > train_progress_tail1.jsonl
ls -1 results/hypertune/trial_00?/train_progress.jsonl > train_progress_tail1_paths.txt
python -c 'import pandas as pd; print(pd.read_json(open("train_progress_tail1.jsonl"), lines=True));' 
