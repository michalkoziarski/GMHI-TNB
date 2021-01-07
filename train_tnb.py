import argparse
import logging
import numpy as np
import pandas as pd

from models import TabNetBaggingClassifier
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score


SEED = 42

DATA_PATH = Path(__file__).parent / 'data'
PROBABILITIES_PATH = Path(__file__).parent / 'probabilities'
RESULTS_PATH = Path(__file__).parent / 'results'


def get_experiment_name(args):
    return '_'.join([f'{k}={v}' for k, v in vars(args).items()])


def get_results_df(args, score):
    columns = list(vars(args).keys()) + ['score']
    row = [vars(args)[column] for column in columns[:-1]] + [score]

    return pd.DataFrame([row], columns=columns)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-fold', type=int, required=True)
    parser.add_argument('-max_features', type=float, default=1.0)
    parser.add_argument('-max_samples', type=float, default=1.0)
    parser.add_argument('-n_estimators', type=int, default=200)

    args = parser.parse_args()

    fold_path = DATA_PATH / f'{args.fold}'
    experiment_name = get_experiment_name(args)

    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    PROBABILITIES_PATH.mkdir(exist_ok=True, parents=True)

    logging.info(f'Running {experiment_name}...')

    X_train = np.load(fold_path / 'X_train.npy')
    y_train = np.load(fold_path / 'y_train.npy')
    X_test = np.load(fold_path / 'X_test.npy')
    y_test = np.load(fold_path / 'y_test.npy')

    clf = TabNetBaggingClassifier(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        max_features=args.max_features,
        random_state=SEED
    )
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    score = balanced_accuracy_score(y_test, predictions)

    logging.info(f'BAC = {score:.4f}')

    probabilities = clf.predict_proba(X_test)
    np.save(PROBABILITIES_PATH / f'{experiment_name}.npy', probabilities)

    df = get_results_df(args, score)
    df.to_csv(RESULTS_PATH / f'{experiment_name}.csv', index=False)
