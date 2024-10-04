import logging
import pickle
import sys

import click
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.append('.')

from configs import MODEL, MODEL_PATH, DS_PATH_SPLIT, MODEL_PARAMS
from utils.utils import load_pickled_file

log = logging.getLogger(__name__)


# import dvc.api

# params = dvc.api.params_show()

# model_params = params['MODEL_PARAMS']


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
    X_train, X_test, y_train, y_test = load_pickled_file(ds_path)
    model = MODEL(**MODEL_PARAMS)
    pipe = train_model(model, X_train, y_train)
    save_model(pipe, model_path)


if __name__ == "__main__":
    train()
