from typing import Tuple

import cv2
import numpy as np

from configs import EPS
from utils.dataset.common import time_of_function


@time_of_function
def foreground_percent(img: np.array) -> float:
    return np.sum(img > 0) / img.size


@time_of_function
def gradients(img: np.array) -> Tuple[float, float, float]:
    foreground = np.sum(img == 255)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = sobelx + sobely
    fx = np.sum(sobelx == 255) / (foreground + EPS)
    fy = np.sum(sobely == 255) / (foreground + EPS)
    fxy1 = np.sum(sobelxy == 255) / (foreground + EPS)
    fxy2 = np.sum(sobelxy == 255 * 2) / (foreground + EPS)
    return fx, fy, fxy1
