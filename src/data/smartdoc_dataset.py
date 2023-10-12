import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from src.data.parsed_image import ParsedImage
from src.settings import N_PROC


def resize(img: np.array, width: int) -> np.array:
    height = int(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, height))


def order_points(pts: np.array) -> np.array:
    """
    Arrange points to start from top left
    """
    pts = pts[:, 0]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def preprocess_image(img: np.array, width: int = 1024, min_page_size=100) -> np.array:
    """
    Prepare image for processing
    """
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize(img, width)
    img_blured = cv2.medianBlur(img, 7)
    thresh = cv2.adaptiveThreshold(
        img_blured,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    # try to find a page on the image
    best_approx, max_rectangle = find_page(thresh)
    # align and crop the page
    if (max_rectangle[2] > min_page_size) and (max_rectangle[3] > min_page_size):
        img = align_page(best_approx, img)
    img = resize(img, width)
    return img


def align_page(best_approx: np.array, img: np.array) -> np.array:
    """
    Align and crop found page
    """
    best_approx = order_points(best_approx)
    pts1 = np.float32(best_approx)
    pts2 = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return img


def find_page(thresh: np.array) -> Tuple[np.array, tuple]:
    """
    Try to find big rectangular object on an image
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_rectangle = (0, 0, 0, 0)
    best_approx = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, thresh.shape[0] // 4, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > max_rectangle[2] * max_rectangle[3]:
                max_rectangle = (x, y, w, h)
                best_approx = approx
    return best_approx, max_rectangle


def get_data_series(file_name: str, finereader_path: Path, tesseract_path: Path, image_folder_path: Path,
                    size: int) -> pd.Series:
    """
    Get feature vector for one document
    """
    img_path = image_folder_path / (file_name + '.jpg')
    finereader_path = finereader_path / (file_name + '.wacc.txt')
    tesseract_path = tesseract_path / (file_name + '.wacc.txt')
    image = cv2.imread(str(img_path))
    image = preprocess_image(image, size)
    parsed_image = ParsedImage(image, file_name)
    with open(tesseract_path, 'r') as tesseract_file:
        tesseract_acc = float(tesseract_file.read().split()[12][:-1])
    with open(finereader_path, 'r') as finereader_file:
        finereader_acc = float(finereader_file.read().split()[12][:-1])
    series = parsed_image.series
    series['tess_acc'] = tesseract_acc
    series['fine_acc'] = finereader_acc
    series = series.astype('float')
    return series


def create_smartdoc_ds(path: Path, save_file: str, size: int = 1024) -> None:
    root_paths = [path / 'Captured_Images' / 'Nokia_phone',
                  path / 'Captured_Images' / 'Samsung_phone']
    images_folder_paths = [p / 'Images' for p in root_paths]
    finereader_paths = [p / 'OCR_Accuracy_Finereader' for p in root_paths]
    tesseract_paths = [p / 'OCR_Accuracy_Tesseract' for p in root_paths]
    file_names = []
    for tesseract_path in tesseract_paths:
        file_names.append(list(x.stem.split('.')[0] for x in list(tesseract_path.glob('*.wacc.txt'))))
    df = []
    for i, image_folder_path in enumerate(images_folder_paths):
        with mp.Pool(N_PROC) as pool:
            df.extend(pool.starmap(get_data_series,
                                   [(file_name, finereader_paths[i], tesseract_paths[i], image_folder_path, size) for
                                    file_name in file_names[i]]))
    df = pd.concat(df, axis=1).T
    with open(save_file, 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    from src.settings import RAW_DATA_PATH
    create_smartdoc_ds(RAW_DATA_PATH, r'data/short_ds.pkl')
