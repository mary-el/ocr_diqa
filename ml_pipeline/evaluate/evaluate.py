import logging
import pickle

import click
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from configs import RESULTS_PATH, MODEL_PATH, DS_PATH_SPLIT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_pickled_file(load_file: str):
    with open(str(load_file), 'rb') as f:
        return pickle.load(f)


def evaluate_model(pipe: Pipeline, X_test: np.array, y_test: np.array):
    y_pred = pipe.predict(X_test)
    metrics = {
        'R2': r2_score,
        'MAE': mean_absolute_error,
        'MSE': mean_squared_error,
        'PLCC': lambda x, y: pearsonr(x, y)[0],
        'SROCC': lambda x, y: spearmanr(x, y, axis=0)[0]
    }
    results = {}
    for name, metric in metrics.items():
        results[name] = metric(y_test, y_pred)
    return y_test, y_pred, results


@click.command()
@click.option("--ds_path", help="Path to preprocessed Smartdoc dataset", default=DS_PATH_SPLIT)
@click.option("--model_path", help="Trained model name", default=MODEL_PATH)
@click.option("--results_file", help="File for saving results", default=RESULTS_PATH)
def evaluate(ds_path: str, model_path: str, results_file: str):
    logger.info(f'Loading model {model_path}')
    _, X_test, _, y_test = load_pickled_file(ds_path)
    model = load_pickled_file(model_path)
    _, _, results = evaluate_model(model, X_test, y_test)
    df = pd.DataFrame.from_dict(results, orient='index')
    logger.info(f'Saving to {results_file}')
    if results_file:
        df.to_csv(results_file, sep=';')
    return results


if __name__ == "__main__":
    evaluate()
