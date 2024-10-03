import logging
import pickle
from typing import Tuple

import click
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from configs import MODEL, MODEL_PARAMS, MODEL_PATH, DS_PATH_SPLIT

log = logging.getLogger(__name__)


def load_split_ds(load_file: str) -> Tuple:
    with open(str(load_file), 'rb') as f:
        return pickle.load(f)


def train_model(model, X_train: np.array, y_train: np.array) -> Pipeline:
    pipe = Pipeline(steps=[
        ('minmax_scaler', MinMaxScaler()),
        ('standard_scaler', StandardScaler()),
        ('regressor', model),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def save_model(pipeline, model_path: str):
    log.info(f'Saving model to {model_path}')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


@click.command()
@click.option("--ds_path", help="Path to preprocessed Smartdoc dataset", default=DS_PATH_SPLIT)
@click.option("--model_path", help="Model name", default=MODEL_PATH)
def train(ds_path: str, model_path: str):
    X_train, X_test, y_train, y_test = load_split_ds(ds_path)
    model = MODEL(**MODEL_PARAMS)
    pipe = train_model(model, X_train, y_train)
    save_model(pipe, model_path)


if __name__ == "__main__":
    train()
