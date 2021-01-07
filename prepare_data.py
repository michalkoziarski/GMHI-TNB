import numpy as np

from data_utils import load_4347, load_val
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


SEED = 42

DATA_PATH = Path(__file__).parent / 'data'
DATA_PATH.mkdir(exist_ok=True, parents=True)

X1, y1 = load_4347(binarize=True)
X2, y2 = load_val()

X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

for i, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), total=10):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    fold_path = DATA_PATH / f'{i}'
    fold_path.mkdir(exist_ok=True, parents=True)

    np.save(fold_path / 'X_train.npy', X_train)
    np.save(fold_path / 'y_train.npy', y_train)
    np.save(fold_path / 'X_test.npy', X_test)
    np.save(fold_path / 'y_test.npy', y_test)
