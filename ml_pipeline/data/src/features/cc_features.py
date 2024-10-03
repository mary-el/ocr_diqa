from typing import Tuple

import numpy as np
import pandas as pd

from configs import EPS
from src.utils.common import time_of_function


@time_of_function
def small_speckle_factor(df: pd.DataFrame, fs: int, min_area=5) -> float:
    filter1 = df['area'] < fs
    filter2 = df['area'] >= min_area
    filter3 = df['area'] < fs * fs
    return len(df[filter1 & filter2]) / (len(df[filter2 & filter3]) + EPS)


@time_of_function
def touching_character_factor(df: pd.DataFrame, fs: int) -> float:
    filter1 = df['h'] / df['w'] < 0.75
    filter2 = df['area'] > 3 * fs
    filter3 = df['h'] > 0.75 * fs
    filter4 = df['h'] < 2 * fs
    return len(df[filter1 & filter2 & filter3 & filter4]) / (len(df[filter2 & filter3 & filter4]) + EPS)


@time_of_function
def white_speckle_factor(df: pd.DataFrame, min_size: int = 3) -> float:
    filter1 = df['h'] < min_size
    filter2 = df['w'] < min_size
    return len(df[filter1 & filter2]) / (len(df) + EPS)


@time_of_function
def small_white_speckle(df_wcc: pd.DataFrame, fs: int, min_size: int = 1) -> float:
    filter1 = df_wcc['area'] > min_size
    filter2 = df_wcc['area'] < 0.02 * fs * fs
    filter3 = df_wcc['area'] < fs * fs
    return len(df_wcc[filter1 & filter2]) / (len(df_wcc[filter1 & filter3]) + EPS)


@time_of_function
def stroke_width(df: pd.DataFrame) -> Tuple[float, float]:
    if len(df) == 0:
        return 0., 0.
    sum_wh = df['w'] + df['h']
    sw = 0.5 * sum_wh + np.sqrt(0.125 * sum_wh * sum_wh - df['area'] / 2)
    return sw.mean(), sw.std()


@time_of_function
def stability_of_cc_values(img: np.array, labels: np.array) -> Tuple[float, float]:
    max_cc_vals, min_cc_vals = [], []
    for i in range(np.max(labels)):
        cc = img[labels == i]
        if np.any(cc):
            max_cc_vals.append(np.max(cc))
            min_cc_vals.append(np.min(cc))
    if len(max_cc_vals) == 0:
        return 0, 0
    return np.std(max_cc_vals), np.std(min_cc_vals)


@time_of_function
def height_width_ratio(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0
    return (df['h'] / df['w']).mean()


@time_of_function
def characters_to_cc_ratio(df_cc: pd.DataFrame, df_char: pd.DataFrame) -> float:
    return len(df_char) / (len(df_cc) + EPS)
