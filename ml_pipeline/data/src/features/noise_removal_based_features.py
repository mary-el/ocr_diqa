from typing import List

import cv2
import numpy as np

from configs import EPS
from src.utils import time_of_function


@time_of_function
def gaussian(img: np.array) -> List:
    blurred = cv2.GaussianBlur(img, (3, 3), 0.95)
    return [np.sum(cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)[1] != img) / (np.sum(img > 0) + EPS) for
            thresh in (120, 150, 180, 190)]


@time_of_function
def median(img: np.array) -> List:
    return [np.sum(img != cv2.medianBlur(img, kernel, 0.95)) / (np.sum(img > 0) + EPS) for kernel in (3, 5)]
