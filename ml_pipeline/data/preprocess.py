import logging
import pickle
import sys

import click
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append('.')
sys.path.append('./ml_pipeline')

from configs import DS_PATH, DS_PATH_SPLIT
from utils.utils import process_df, dump_to_pickle_file

log = logging.getLogger(__name__)


@click.command()
@click.option("--ds_path", help="Path to the CSV with data", default=DS_PATH)
@click.option("--save_file", help="File for saving split dataset", default=DS_PATH_SPLIT)
def preprocess_dataset(ds_path: str, save_file: str) -> None:
    '''
    Splitting and cleaning dataset
    '''
    log.info(f'Loading {ds_path}')
    df = pd.read_csv(ds_path, index_col=0)
    df.drop(columns=['tess_acc'], inplace=True)
    arr = process_df(df)
    X, y = arr[:, :-1], arr[:, -1]
    y = y / 100.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info(f'Saving to {save_file}')
    dump_to_pickle_file(save_file, (X_train, X_test, y_train, y_test))


if __name__ == "__main__":
    preprocess_dataset()
