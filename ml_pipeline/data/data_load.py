import sys
from pathlib import Path

import click

sys.path.append('.')

from configs import RAW_DATA_PATH, DS_PATH
from utils.dataset.smartdoc_dataset import create_smartdoc_ds

import urllib.request
import zipfile
from configs import ZIP_PATH, DS_LINK


def download_smartdoc_ds(data_path):
    urllib.request.urlretrieve(DS_LINK, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(data_path)


@click.command()
@click.option("--ds_path", help="Path to the Smartdoc dataset", default=RAW_DATA_PATH)
@click.option("--save_to", help="Path to csv file with parsed dataset", default=DS_PATH)
@click.option("--download", help="Download file from link", is_flag=True)
def data_load(ds_path: str, save_to: str, size: int, download: bool):
    if download:
        download_smartdoc_ds(ds_path)
    create_smartdoc_ds(Path(ds_path), save_to, size)


if __name__ == "__main__":
    data_load()
