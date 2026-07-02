import pandas as pd
import sys

def main(*args):
    if not len(args):
	args.append(next(iter(Path('results').glob('hypertune-large-24-26/trial_*/train_progress.jsonl'))))
    columns = args[1:]
    df = pd.read_json(args[0], lines=True)
    columns = ['epoch', 'learning_rate', 'valid_action_accuracy', 'valid_route_accuracy'] + [c for c in columns if c in df.columns]
    print_df(df)


def chunk_df(df, rows=min(10, pd.options.display.max_rows)):
    """ Use slicing to create a one-liner """
    chunks = []
    for i in range(1 + len(df) / rows):
	chunks.append(df.iloc[i*rows:i*rows + rows + 1])
    return chunks


def print_df(df, rows=min(10, pd.options.display.max_rows)):
    for chunk in chunk_df(df, rows=rows):
        print(chunk)


if __name__ == '__main__':
    return main(args=sys.argv[1:])
