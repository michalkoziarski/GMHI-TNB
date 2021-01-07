import numpy as np

from dataclasses import dataclass
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass
class TabNetBaggingClassifier(ClassifierMixin, BaseEstimator):
    n_estimators: int = 200
    max_samples: float = 1.0
    max_features: float = 0.5
    bootstrap: bool = False
    oob_score: bool = False
    max_epochs: int = 500
    patience: int = 200
    random_state: int = None
    verbose: int = 0
    device_name: str = 'auto'

    def __post_init__(self):
        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.estimators_ = []
        self.features_ = []
        self.classes_ = np.unique(y)
        self.n_samples_ = int(np.round(self.max_samples * X.shape[0]))
        self.n_features_ = int(np.round(self.max_features * X.shape[1]))

        for _ in range(self.n_estimators):
            samples = np.random.choice(X.shape[0], size=self.n_samples_, replace=self.bootstrap)
            features = np.random.choice(X.shape[1], size=self.n_features_, replace=False)

            unused_samples = np.array([i for i in range(X.shape[0]) if i not in samples])

            X_train = X[samples][:, features]
            y_train = y[samples]

            estimator = TabNetClassifier(
                verbose=self.verbose,
                device_name=self.device_name
            )

            if self.oob_score and len(unused_samples) > 0:
                X_val = X[unused_samples][:, features]
                y_val = y[unused_samples]

                estimator.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=['balanced_accuracy'],
                    patience=self.patience,
                    max_epochs=self.max_epochs
                )
            else:
                estimator.fit(
                    X_train, y_train,
                    patience=self.patience,
                    max_epochs=self.max_epochs
                )

            self.estimators_.append(estimator)
            self.features_.append(features)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['estimators_', 'features_', 'classes_'])

        X = check_array(X)

        predictions = []

        for estimator, features in zip(self.estimators_, self.features_):
            predictions.append(estimator.predict_proba(X[:, features]))

        return np.mean(predictions, axis=0)

    def predict(self, X):
        predicted_probability = self.predict_proba(X)

        return self.classes_.take((np.argmax(predicted_probability, axis=1)), axis=0)
