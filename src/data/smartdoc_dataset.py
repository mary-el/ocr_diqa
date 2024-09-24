import multiprocessing as mp
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm

from src.data.parsed_image import ParsedImage
from src.settings import N_PROC
from src.utils.preprocess_smartdoc import preprocess_image


def get_data_series(file_name: str, finereader_path: Path, tesseract_path: Path, image_folder_path: Path,
                    size: int) -> pd.Series:
    """
    Get feature vector for one document
    """
    img_path = image_folder_path / (file_name + '.jpg')
    finereader_path = finereader_path / (file_name + '.wacc.txt')
    tesseract_path = tesseract_path / (file_name + '.wacc.txt')
    features = get_features(img_path, size)
    with open(tesseract_path, 'r') as tesseract_file:
        tesseract_acc = float(tesseract_file.read().split()[12][:-1])
    with open(finereader_path, 'r') as finereader_file:
        finereader_acc = float(finereader_file.read().split()[12][:-1])
    features['tess_acc'] = tesseract_acc
    features['fine_acc'] = finereader_acc
    features = features.astype('float')
    return features


def get_features(img_path, size, save_prepared: bool = True, rotate: bool = True):
    image = cv2.imread(str(img_path))
    image = preprocess_image(image, size, rotate=rotate)
    file_name = img_path.name
    if save_prepared:
        cv2.imwrite('data/preprocessed/' + file_name + '.jpg', image)
    parsed_image = ParsedImage(image, file_name)
    return parsed_image.series


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
                                   tqdm.tqdm(
                                       [(file_name, finereader_paths[i], tesseract_paths[i], image_folder_path, size)
                                        for file_name in file_names[i]])))
    df = pd.concat(df, axis=1).T
    with open(save_file, 'wb') as f:
        pickle.dump(df, f)


def create_eval_ds(path: Path, save_file: str, size: int = 1024) -> None:
    files = list(path.glob('*.jpeg'))
    df = [get_features(img_path, size, rotate=False) for img_path in files]
    df = np.stack(df)
    with open(save_file, 'wb') as f:
        pickle.dump(df, f)
