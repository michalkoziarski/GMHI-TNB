import logging
import numpy as np
import pandas as pd
import pickle

from data_utils import load_4347, load_val
from mlxtend.classifier import EnsembleVoteClassifier
from models import TabNetBaggingClassifier
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 42

DATA_PATH = Path(__file__).parent / 'data'
MODELS_PATH = Path(__file__).parent / 'models'
PROBABILITIES_PATH = Path(__file__).parent / 'probabilities'
RESULTS_PATH = Path(__file__).parent / 'results'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    MODELS_PATH.mkdir(exist_ok=True, parents=True)
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    PROBABILITIES_PATH.mkdir(exist_ok=True, parents=True)

    X_train, y_train = load_4347(binarize=True)
    X_test, y_test = load_val(exclude_weight=True)

    classifiers = {
        'TNB': Pipeline([
            ('scaler', StandardScaler()),
            ('tnb', TabNetBaggingClassifier(random_state=SEED))
        ]),
        'RF': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=1000, random_state=SEED))
        ]),
        'GB': Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingClassifier(n_estimators=1000, random_state=SEED))
        ]),
        'XGB': Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(n_estimators=1000, random_state=SEED))
        ])
    }

    results = []

    for classifier_name, clf in classifiers.items():
        logging.info(f'Training {classifier_name}...')

        clf.fit(X_train, y_train)

        score = balanced_accuracy_score(y_test, clf.predict(X_test))

        logging.info(f'{classifier_name} BAC = {score:.4f}')

        probabilities = clf.predict_proba(X_test)
        np.save(PROBABILITIES_PATH / f'{classifier_name}.validation.npy', probabilities)

        results.append([classifier_name, score])

    ensemble = EnsembleVoteClassifier(list(classifiers.values()), voting='soft', fit_base_estimators=False)
    ensemble.fit(X_train, y_train)

    score = balanced_accuracy_score(y_test, ensemble.predict(X_test))

    logging.info(f'Ensemble BAC = {score:.4f}')

    results.append(['Ensemble', score])

    with open(MODELS_PATH / f'ensemble.4347.pickle', 'wb') as f:
        pickle.dump(ensemble, f)

    df = pd.DataFrame(results, columns=['Classifier', 'BAC'])
    df.to_csv(RESULTS_PATH / f'validation.csv', index=False)
