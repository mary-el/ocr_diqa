import logging
from typing import Callable, Dict, List, Optional

import cv2 as cv

try:
    import doxapy
except ImportError:
    logging.warning("Cannot use doxa binarization function because doxa library is not installed")

import numpy as np
from pylsd import lsd
from skimage.exposure import exposure
from skimage.filters import threshold_sauvola

from src.utils.common_helpers import whitening_lines


def smart_whitening_lines(
        image: np.ndarray,
        v_thresh: float = 0.07,
        h_thresh: float = 0.05,
        v_restore: bool = True,
        h_restore: bool = True,
        iterations: int = 1,
) -> np.ndarray:
    image, vertical_mask, horizontal_mask = whitening_lines(
        image, vertical_threshold=v_thresh, horizontal_threshold=h_thresh
    )
    if v_thresh > 0 and v_restore:
        vertical_mask = vertical_mask.astype("uint8") * 255
        black_lines_mask = cv.threshold(vertical_mask, 0, 255, cv.THRESH_BINARY_INV)[1]
        tresh_lines = correct_tresh_lines(black_lines_mask, 0.01, 0, 1, iterations=iterations)
        vertical_mask_bool = tresh_lines == 0
        image[vertical_mask_bool] = 0

    if h_thresh > 0 and h_restore:
        horizontal_mask = horizontal_mask.astype("uint8") * 255
        black_lines_mask = cv.threshold(horizontal_mask, 0, 255, cv.THRESH_BINARY_INV)[1]
        tresh_lines = correct_tresh_lines(black_lines_mask, 0, 0.01, 1, iterations=iterations)
        vertical_mask_bool = tresh_lines == 0
        image[vertical_mask_bool] = 0

    return image


def correct_tresh_lines(
        black_lines_mask,
        vertical_ratio: float = 0,
        horizontal_ratio: float = 0,
        k_thickness: float = 1,
        iterations: int = 1,
) -> np.ndarray:
    h, w = black_lines_mask.shape[:2]
    horizontal_threshold = w * horizontal_ratio
    vertical_threshold = h * vertical_ratio

    for _ in range(0, iterations):
        lines = lsd(img=black_lines_mask, quant=0.005, ang_th=45)
        for element in lines:
            x1 = int(element[0])
            x2 = int(element[2])
            y1 = int(element[1])
            y2 = int(element[3])
            # If the length of the line is more than threshold, then draw a white line on it
            if abs(x1 - x2) > horizontal_threshold or abs(y1 - y2) > vertical_threshold:
                width = element[4] / k_thickness
                # Draw the white line
                cv.line(
                    black_lines_mask,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    int(np.ceil(width)),
                )

    return black_lines_mask


def detect_text_line_height(image):
    # Calculate window parameter from OpenCV line's heght

    # Edge detection
    dst = cv.Canny(image, 50, 150, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    red_lines = np.zeros_like(cdstP)

    # Probabilistic Line Transform
    minLineLength = 80
    maxLineGap = 25
    linesP = cv.HoughLinesP(
        image=dst,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        lines=np.array([]),
        minLineLength=minLineLength,
        maxLineGap=maxLineGap,
    )

    # Draw the lines into temp array
    if linesP is not None:
        for i in range(0, len(linesP)):
            # line = linesP[i][0]
            # cv.line(red_lines, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv.FILLED)
            x1, y1, x2, y2 = linesP[i][0]
            if np.arctan2(abs(y2 - y1), x2 - x1) * 180.0 / np.pi <= 1:
                cv.line(
                    red_lines, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.FILLED
                )

    img_h = red_lines.shape[0]
    img_w = red_lines.shape[1]
    lines_h = []
    inrowcount = 0
    mean_lines_heigth = None

    # Detect mean line's height in ten slices
    for i in range(0, img_w, round(img_w / 30)):
        for px in range(img_h):
            if red_lines[px, i, 2] != 0:
                inrowcount += 1
            else:
                if inrowcount != 0:
                    lines_h.append(inrowcount)
                inrowcount = 0

    if lines_h:
        mean_lines_heigth = round(np.array(lines_h).mean())
        std_lines_heigth = round(np.array(lines_h).std())
        if std_lines_heigth != 0:
            stroke_h_clear = [
                stroke_heigth
                for stroke_heigth in lines_h
                if abs(stroke_heigth - mean_lines_heigth) / std_lines_heigth < 1.3
            ]
        else:
            stroke_h_clear = lines_h
        mean_lines_heigth = round(np.array(stroke_h_clear).mean())

    return mean_lines_heigth or -1


def __sauvola_binarization_lut(image: np.ndarray) -> np.ndarray:
    # Adjust contrast
    image = white_image_correction(image)
    window_size = get_window_size(image)

    # Sauvola binarization
    thresh_sauvola = threshold_sauvola(image, window_size=window_size)

    binary_sauvola = image > thresh_sauvola
    binary_sauvola = binary_sauvola.astype("uint8") * 255

    return binary_sauvola


def white_image_correction(image: np.ndarray) -> np.ndarray:
    if is_white(image):
        # contrast by LUT
        gamma = 1.5
        look_up_table = np.empty((1, 256), np.uint8)
        for i in range(256):
            look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        image = cv.LUT(image, look_up_table)
    return image


def is_white(image: np.ndarray) -> bool:
    empty_threshold = 0.003
    contrast_threshold = 0.22
    empty_list = []
    slice_count = 10
    slice_start_point = 0
    slice_length = round(image.shape[0] / slice_count)
    slice_end_point = slice_start_point + slice_length
    is_image_low_contrast = exposure.is_low_contrast(
        image,
        fraction_threshold=contrast_threshold,
        lower_percentile=5,
        upper_percentile=95,
    )
    # Checking if most of image is white
    for _ in range(slice_count):
        cur_img_part = image[slice_start_point:slice_end_point]
        slice_start_point += slice_length

        if slice_start_point + slice_length < image.shape[0]:
            slice_end_point = slice_start_point + slice_length
        else:
            slice_end_point = image.shape[0]
            break

        hist = cv.calcHist([cur_img_part], [0], None, [2], [0, 256])

        if hist[0] / (hist[1] + 0.000001) < empty_threshold:
            empty_list.append(0)
        else:
            empty_list.append(1)

    return bool(np.array(empty_list).mean() >= 0.5 and is_image_low_contrast)


def get_window_size(image: np.ndarray) -> int:
    text_line_height = detect_text_line_height(image)
    window_size = round(text_line_height * 1.8)
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(window_size, 25)
    return window_size


def __doxa_binarization(image: np.ndarray, binarizator) -> np.ndarray:
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binarized_image = np.empty(image.shape, image.dtype)
    image = white_image_correction(image)
    window_size = get_window_size(image)

    binarizator.initialize(image)
    k_size = -0.15  # Default value
    binarizator.to_binary(binarized_image, {"window": window_size, "k": k_size})

    return binarized_image
