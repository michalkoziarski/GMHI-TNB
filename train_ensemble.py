import argparse
import logging
import numpy as np
import pandas as pd

from models import TabNetBaggingClassifier
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier


SEED = 42

DATA_PATH = Path(__file__).parent / 'data'
MODELS_PATH = Path(__file__).parent / 'models'
PROBABILITIES_PATH = Path(__file__).parent / 'probabilities'
RESULTS_PATH = Path(__file__).parent / 'results'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-fold', type=int, required=True)

    args = parser.parse_args()

    fold_path = DATA_PATH / f'{args.fold}'

    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    PROBABILITIES_PATH.mkdir(exist_ok=True, parents=True)

    X_train = np.load(fold_path / 'X_train.npy')
    y_train = np.load(fold_path / 'y_train.npy')
    X_test = np.load(fold_path / 'X_test.npy')
    y_test = np.load(fold_path / 'y_test.npy')

    classifiers = {
        'TNB': TabNetBaggingClassifier(random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=1000, random_state=SEED),
        'GB': GradientBoostingClassifier(n_estimators=1000, random_state=SEED),
        'XGB': XGBClassifier(n_estimators=1000, random_state=SEED)
    }

    results = []

    for classifier_name, clf in classifiers.items():
        logging.info(f'Training {classifier_name}...')

        clf.fit(X_train, y_train)

        score = balanced_accuracy_score(y_test, clf.predict(X_test))

        logging.info(f'{classifier_name} BAC = {score:.4f}')

        probabilities = clf.predict_proba(X_test)
        np.save(PROBABILITIES_PATH / f'{classifier_name}_{args.fold}.npy', probabilities)

        results.append([classifier_name, score])

    ensembles = {
        'VCS': VotingClassifier(list(classifiers.items()), voting='soft'),
        'VCH': VotingClassifier(list(classifiers.items()), voting='hard')
    }

    for classifier_name, clf in ensembles.items():
        logging.info(f'Training {classifier_name}...')

        clf.fit(X_train, y_train)

        score = balanced_accuracy_score(y_test, clf.predict(X_test))

        logging.info(f'{classifier_name} BAC = {score:.4f}')

        results.append([classifier_name, score])

    df = pd.DataFrame(results, columns=['Classifier', 'BAC'])
    df.to_csv(RESULTS_PATH / f'{args.fold}.csv', index=False)
