from dataclasses import dataclass
from pathlib import Path
from typing import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.features.cc_features import (
    small_speckle_factor,
    touching_character_factor,
    broken_character_factor,
    white_speckle_factor,
    stroke_width,
    small_white_speckle,
    height_width_ratio,
    characters_to_cc_ratio
)
from src.data.features.morph_based_features import erosion, closing, opening, gaussian
from src.data.features.noise_removal_based_features import gaussian, median
from src.data.features.spatial_characteristic_features import (
    foreground_percent,
    gradients
)
from src.data.features.statistical_features import (
    mean_sd,
    grounds_mean,
)


def get_cc(img: np.array) -> Tuple[pd.DataFrame, np.array]:
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    ccs = []
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        ccs.append((i, x, y, w, h, area))
    df = pd.DataFrame(ccs, columns=["label", "x", "y", "w", "h", "area"]).dropna()
    return df, labels


class Features:
    pass


@dataclass
class ParsedImage:
    width: int
    path: Path
    image: np.array
    thresh: np.array
    inversed: np.array
    df_cc: pd.DataFrame
    labels_cc: np.array
    df_wcc: pd.DataFrame
    labels_wcc: np.array
    df_char: pd.DataFrame
    labels_filtered: np.array
    fs: int
    features: Features
    series: pd.Series

    def _resize(self, img: np.array) -> np.array:
        height = int(img.shape[0] * self.width / img.shape[1])
        return cv2.resize(img, (self.width, height))

    def show_image(self) -> None:
        plt.imshow(self.image, cmap="gray")
        plt.waitforbuttonpress()

    def show_binarized(self) -> None:
        plt.imshow(self.thresh, cmap="gray")
        plt.waitforbuttonpress()

    def show_char_labels(self) -> None:
        plt.imshow(self.labels_filtered, cmap="prism")
        plt.waitforbuttonpress()

    def get_threshold(self, adaptive: bool = True) -> np.array:
        if adaptive:
            img_blured = cv2.medianBlur(self.image, 3)
            thresh = cv2.adaptiveThreshold(
                img_blured,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
        else:
            thresh = cv2.threshold(
                self.image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
        return thresh

    def get_inversed(self) -> np.array:
        return cv2.bitwise_not(self.thresh)

    def get_bcc(self) -> Tuple[pd.DataFrame, np.array]:
        """
        Get Black Connected Components
        """
        df, labels = get_cc(self.thresh)
        return df, labels

    def get_wcc(self) -> Tuple[pd.DataFrame, np.array]:
        """
        Get White Connected Components
        """
        df, labels = get_cc(self.inversed)
        return df, labels

    def filter_cc(self, min_height=5, max_height=20, min_width=2, max_width=100) -> Tuple[pd.DataFrame, np.array]:
        """
        Filter character-sized connected components
        """
        min_area = min_height * min_width
        filter1 = self.df_cc["h"] >= min_height
        filter2 = self.df_cc["h"] <= max_height
        filter3 = self.df_cc["w"] >= min_width
        filter4 = self.df_cc["w"] <= max_width
        filter5 = self.df_cc["area"] >= min_area
        df_char = self.df_cc[filter1 & filter2 & filter3 & filter4 & filter5]
        keep_labels = df_char["label"].to_list()
        labels_filtered = np.copy(self.labels_cc)
        labels_filtered[~np.isin(labels_filtered, keep_labels)] = 0
        return df_char, labels_filtered

    def get_font_size(self) -> int:
        if len(self.df_char) == 0:
            return 0
        return self.df_char["h"].median()

    def get_feature_df(self) -> pd.Series:
        feature_names, feature_vals = [], []
        for category_name, category in self.features.__dict__.items():
            for feature_name, feature in category.__dict__.items():
                new_names, new_vals = [category_name + '.' + feature_name], [feature]
                if isinstance(feature, Sequence):
                    new_names = [f'{new_names[0]}_{i}' for i in range(1, len(feature) + 1)]
                    new_vals = feature
                feature_names.extend(new_names)
                feature_vals.extend(new_vals)
        return pd.Series(feature_vals, index=feature_names)

    def __init__(self, image: np.array, name: str) -> None:
        import time
        start_time = time.perf_counter()
        self.name = name
        self.image = image
        self.width = image.shape[0]
        self.thresh = self.get_threshold()
        self.inversed = self.get_inversed()
        self.df_cc, self.labels_cc = self.get_bcc()
        self.df_wcc, self.labels_wcc = self.get_wcc()
        self.df_char, self.labels_filtered = self.filter_cc()
        self.fs = self.get_font_size()
        self.features = Features(self)
        self.series = self.get_feature_df()
        print(f'{time.perf_counter() - start_time:.4f}')


@dataclass
class Features:
    def __init__(self, image: ParsedImage) -> None:
        self.cc = ConnectedComponentsFeatures(image)
        self.morph = MorphologicalBasedFeatures(image)
        self.noise = NoiseRemovalBasedFeatures(image)
        self.spatial = SpatialCharacteristicFeatures(image)
        self.statistical = StatisticalFeatures(image)


class ConnectedComponentsFeatures:
    def __init__(self, image: ParsedImage) -> None:
        self.SSF = small_speckle_factor(image.df_cc, image.fs)
        self.TCF = touching_character_factor(image.df_cc, image.fs)
        self.WSF = white_speckle_factor(image.df_wcc)
        self.SWS = small_white_speckle(image.df_wcc, image.fs)
        self.BCF = broken_character_factor(image.df_cc, image.fs)
        self.SW = stroke_width(image.df_cc)
        # self.stability = stability_of_cc_values(image.image, image.labels_cc)
        self.height_width_ratio = height_width_ratio(image.df_char)
        self.characters_to_cc_ratio = characters_to_cc_ratio(image.df_cc, image.df_char)


class MorphologicalBasedFeatures:
    def __init__(self, image: ParsedImage) -> None:
        self.erosion = erosion(image.thresh)
        self.dilation = erosion(image.inversed)
        self.closing = closing(image.thresh)
        self.opening = opening(image.thresh)


class NoiseRemovalBasedFeatures:
    def __init__(self, image: ParsedImage) -> None:
        self.gaussian = gaussian(image.thresh)
        self.median = median(image.thresh)


class SpatialCharacteristicFeatures:
    def __init__(self, image: ParsedImage) -> None:
        self.foreground_percent = foreground_percent(image.thresh)
        self.gradients = gradients(image.thresh)
        # self.length_of_segment = length_of_segment(image.thresh)
        # self.lps_si = lpc_si(image.image)


class StatisticalFeatures:
    def __init__(self, image: ParsedImage) -> None:
        # self.entropy = image_entropy(image.image)
        self.mean_sd = mean_sd(image.image)
        self.grounds_mean = grounds_mean(image.image, image.thresh)
        # self.uniformities = uniformities(image.image, image.thresh)
