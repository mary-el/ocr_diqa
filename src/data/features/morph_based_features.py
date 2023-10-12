import cv2
import numpy as np

from src.settings import EPS
from src.utils import time_of_function


@time_of_function
def erosion(img: np.array, max_iter: int = 250) -> float:
    kernel = np.ones((3, 3), np.uint8)
    img_er = img
    for i in range(max_iter):
        img_er = cv2.erode(img_er, kernel)
        if np.min(img_er) == np.max(img_er):
            return i / max_iter
    return 1.


@time_of_function
def closing(img: np.array) -> float:
    kernel = np.ones((3, 3), np.uint8)
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    closing = np.sum(img != img_closed) / (np.sum(img > 0) + EPS)
    return closing


@time_of_function
def opening(img: np.array) -> float:
    kernel = np.ones((3, 3), np.uint8)
    img_closed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = np.sum(img != img_closed) / (np.sum(img > 0) + EPS)
    return closing


@time_of_function
def gaussian(img: np.array) -> float:
    blurred = cv2.GaussianBlur(img, (3, 3), 0.95)
    return np.sum(img != blurred) / (np.sum(img > 0) + EPS)
