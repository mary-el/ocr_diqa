from typing import Tuple

import cv2
import numpy as np
from numpy.fft import fft2, ifft2

from src.data.features.utils.libsmop import length, arange
from src.data.features.utils.logGabor_2D import logGabor_2D
from src.settings import EPS
from src.utils import time_of_function


@time_of_function
def foreground_percent(img: np.array) -> float:
    return np.sum(img > 0) / img.size


@time_of_function
def gradients(img: np.array) -> Tuple[float, float, float, float]:
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
    return fx, fy, fxy1, fxy2


@time_of_function
def lpc_si(im: np.array, C=2, Beta_k=0.0001, scales=(1, 3 / 2, 2), w=(1, -3, 2), norient=8) -> float:
    nscale = length(scales)
    row, col = im.shape
    sigma = 0.33

    B = int((np.minimum(row, col) / 16).round())

    gabor_filter = logGabor_2D(im, norient, nscale, scales, sigma)
    imfft = fft2(im)
    s_lpc = np.ones((row, col, norient))

    M = np.zeros((*gabor_filter.shape[2:], gabor_filter.shape[0]))
    energy = np.zeros_like(s_lpc)
    for o in range(norient):
        for s in range(nscale):
            M[:, :, s] = ifft2(np.multiply(imfft, gabor_filter[s, o])).real
            s_lpc[:, :, o] = np.multiply(s_lpc[:, :, o], M[:, :, s] ** w[s])

        e = np.abs(M[:, :, 1])
        e_center = e[B:-B, B:-B]
        e_mean = np.mean(np.ravel(e_center))
        e_std = np.std(np.ravel(e_center))
        T = e_mean + 2 * e_std
        e = np.maximum(0, e - T)
        energy[:, :, o] = e

    s_lpc_map = np.cos(np.angle(s_lpc))
    s_lpc_map[s_lpc_map < 0] = 0
    lpc_map = (np.sum(np.multiply(s_lpc_map, energy), axis=2)) / (
            np.sum(energy, axis=2) + C
    )
    lpc_map_center = lpc_map[B:-B, B:-B]

    sorted_si = np.sort(np.ravel(lpc_map_center))[::-1].T
    N = length(sorted_si)
    u = np.exp(-((arange(0, (N - 1))) / (N - 1)) / Beta_k)
    si = np.sum(np.multiply(sorted_si, u)) / np.sum(u)

    return si
