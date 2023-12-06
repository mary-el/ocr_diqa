from typing import Tuple

import cv2
import numpy as np
from scipy.stats import entropy

from src.settings import EPS
from src.utils.common import time_of_function


@time_of_function
def image_entropy(img: np.array) -> Tuple:
    entropies = []
    for bins in (2 ** i for i in range(1, 8)):
        hist, _ = np.histogram(img.ravel(), bins=bins)
        prob_dist = hist / (hist.sum() + EPS)
        entropies.append(entropy(prob_dist, base=2))
    return tuple(entropies)


@time_of_function
def mean_sd(img: np.array) -> Tuple:
    return tuple(map(lambda x: float(x[0][0]), cv2.meanStdDev(img)))


def grounds_mean(orig_img: np.array, thresh: np.array) -> Tuple[float, float]:
    if not np.any(thresh > 0):
        return 0., np.mean(orig_img)
    return np.mean(orig_img[thresh > 0]), np.mean(orig_img[thresh == 0])


@time_of_function
def uniformities(img: np.array, thresh: np.array) -> Tuple[float, float]:
    foreground_means, background_means = [], []
    for i in (0, 2, 4):
        parts = 2 ** i
        part_size = img.shape[0] // parts
        for j in range(0, part_size * parts, part_size):
            foreground_mean, background_mean = grounds_mean(img[j: j + part_size], thresh[j: j + part_size])
            single_peak_threshold = (background_mean - foreground_mean) / foreground_mean / (255 - background_mean)
            if single_peak_threshold >= 0.01:
                foreground_means.append(foreground_mean)
                background_means.append(background_mean)
    return np.std(foreground_means), np.std(background_means)
