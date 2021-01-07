import numpy as np
import pandas as pd

from pathlib import Path


INPUT_PATH = Path(__file__).parent / 'input'


def get_species(exclude_unclassified_and_viruses=True):
    df_4347 = pd.read_csv(INPUT_PATH / '4347_final_relative_abundances.txt', sep='\t')
    df_val = pd.read_csv(INPUT_PATH / 'validation_abundance.csv')

    if exclude_unclassified_and_viruses:
        df_4347 = df_4347[~df_4347['Unnamed: 0'].str.contains('unclassified')]
        df_4347 = df_4347[~df_4347['Unnamed: 0'].str.contains('virus')]

        df_val = df_val[~df_val['Species'].str.contains('unclassified')]
        df_val = df_val[~df_val['Species'].str.contains('virus')]

    s1 = list(df_4347.transpose().iloc[0])
    s2 = list(df_val.transpose().iloc[0])

    return sorted(list(set(s1 + s2)))


def get_classes():
    df = pd.read_csv(INPUT_PATH / '4347_final_relative_abundances.txt', sep='\t')

    return list(set([x.split('_')[0] for x in df.columns[1:]]))


def get_class_dict(classes=None):
    if classes is None:
        classes = get_classes()

    return {k: v for v, k in enumerate(['Healthy'] + [x for x in classes if x != 'Healthy'])}


def load_4347(species=None, class_dict=None, binarize=False):
    if species is None:
        species = get_species()

    if class_dict is None:
        class_dict = get_class_dict()

    df = pd.read_csv(INPUT_PATH / '4347_final_relative_abundances.txt', sep='\t')
    df = df.set_index('Unnamed: 0').T

    X = np.empty((len(df), len(species)), dtype=np.float32)
    y = np.empty(len(df), dtype=np.uint64)

    i = 0

    for index, row in df.iterrows():
        y[i] = class_dict[index.split('_')[0]]

        for j, s_j in enumerate(species):
            X[i, j] = row.get(s_j, 0.0)

        i += 1

    if binarize:
        y[y > 1] = 1

    return X, y


def load_val(species=None):
    if species is None:
        species = get_species()

    df = pd.read_csv(INPUT_PATH / 'validation_abundance.csv')
    df = df.set_index('Species').T

    md = pd.read_csv(INPUT_PATH / 'validation_metadata.csv')

    X = np.empty((len(df), len(species)), dtype=np.float32)
    y = np.empty(len(df), dtype=np.uint64)

    i = 0

    for index, row in df.iterrows():
        mds = md[md['Sample_Ids'] == index]

        assert len(mds) == 1

        label = mds.iloc[0]['Phenotype']

        if label == 'Healthy':
            y[i] = 0
        elif label == 'Unhealthy':
            y[i] = 1
        else:
            raise ValueError

        for j, s_j in enumerate(species):
            X[i, j] = row.get(s_j, 0.0)

        i += 1

    return X, y
