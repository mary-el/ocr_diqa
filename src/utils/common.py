import logging
import sys
from contextlib import contextmanager
from time import perf_counter

import cv2
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)


def time_of_function(function):
    def wrapped(*args):
        start_time = perf_counter()
        res = function(*args)
        logging.info(f'{function.__name__}: {perf_counter() - start_time:.4f}')
        return res

    return wrapped


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f'Time: {perf_counter() - start:.3f} seconds')


def resize(img: np.array, width: int) -> np.array:
    height = int(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, height))
