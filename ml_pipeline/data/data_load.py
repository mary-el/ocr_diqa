from pathlib import Path

import click

from configs import RAW_DATA_PATH, DS_PATH
from src.download_ds import download_smartdoc_ds
from src.smartdoc_dataset import create_smartdoc_ds


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
