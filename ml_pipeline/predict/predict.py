import logging
import sys

import click

sys.path.append('.')

from configs import RESULTS_PATH, MODEL_PATH, SIZE
from utils.utils import load_pickled_file, process_df

from utils.dataset.smartdoc_dataset import get_features

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option("--model_path", help="Trained model name", default=MODEL_PATH)
@click.option("--image_path", help="Path to the image", default=RESULTS_PATH)
def predict(model_path: str, image_path: str):
    logger.info(f'Loading model {model_path}')
    model = load_pickled_file(model_path)
    df = get_features(image_path, size=SIZE, save_prepared=False).to_frame().transpose()
    X = process_df(df)
    result = model.predict(X)
    logger.info(f'Quality: {result}')
    return result


if __name__ == "__main__":
    predict()
