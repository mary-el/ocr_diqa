import pickle

import numpy as np


def load_pickled_file(load_file: str):
    with open(str(load_file), 'rb') as f:
        return pickle.load(f)


def process_df(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    return np.array(df)
