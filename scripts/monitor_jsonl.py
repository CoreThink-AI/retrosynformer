#!/usr/bin/env python3
"""Tail results/train_progress.jsonl and print a formatted progress table."""
import json
import sys
import time

path = sys.argv[1] if len(sys.argv) > 1 else "results/train_progress.jsonl"

print(f"{'ep':>4}  {'train':>7}  {'valid':>7}  {'route':>7}  {'loss':>10}  {'secs':>5}")
print("-" * 52)

seen = 0
while True:
    with open(path) as f:
        lines = f.readlines()
    for line in lines[seen:]:
        r = json.loads(line)
        print(f"{r['epoch']:>4}  {r['train_action_accuracy']*100:>6.2f}%  {r['valid_action_accuracy']*100:>6.2f}%  {r['valid_route_accuracy']*100:>6.2f}%  {r['valid_loss']:>10.6f}  {r['seconds_per_epoch']:>4.0f}s")
    seen = len(lines)
    time.sleep(10)
