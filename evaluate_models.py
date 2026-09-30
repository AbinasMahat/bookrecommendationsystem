"""Create cached leave-one-out ranking metrics for the evaluation dashboard."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / 'evaluation_metrics.json'
TOP_K = 8


def main():
    ratings_path = BASE_DIR / 'Ratings.csv'
    if not ratings_path.exists():
        ratings_path = BASE_DIR / 'ratings.csv'
    ratings = pd.read_csv(ratings_path)
    ratings.columns = ratings.columns.str.strip()
    ratings = ratings.rename(columns={'Book-Rating': 'rating', 'User-ID': 'user_id'})
    required = {'user_id', 'ISBN', 'rating'}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f'Missing rating columns: {sorted(missing)}')

    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce').fillna(0)
    ratings = ratings[ratings['rating'] > 0].copy()
    holdout = ratings.sort_values('user_id').groupby('user_id').tail(1)
    training = ratings.drop(index=holdout.index)
    popular_items = training.groupby('ISBN')['rating'].agg(['count', 'mean'])
    popular_items['score'] = popular_items['count'] * popular_items['mean']
    ranked_items = popular_items.sort_values(['score', 'mean'], ascending=False).index.astype(str).tolist()

    precisions = []
    recalls = []
    ndcgs = []
    ranked_items = set(ranked_items[:TOP_K])
    for _, row in holdout.iterrows():
        target = str(row['ISBN'])
        hit = target in ranked_items
        precisions.append(1 / TOP_K if hit else 0)
        recalls.append(1.0 if hit else 0.0)
        ndcgs.append(1.0 if hit else 0.0)

    def average(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    payload = {
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'protocol': 'Leave-one-out popularity baseline, top 8',
        'sample_count': int(len(holdout)),
        'metrics': {
            'precision_at_8': average(precisions),
            'recall_at_8': average(recalls),
            'ndcg_at_8': average(ndcgs),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
