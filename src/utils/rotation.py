from pathlib import  Path
from typing import  Union

import cv2 as cv
import numpy as np
import pytesseract


from rotation_helpers import (
    only_text_angle,
    rotate_image_small,
    rotate_image
)


def rotate_origin_image(image: Union[str, Path, np.ndarray]):
    if not type(image) == np.ndarray:
        image = cv.imread(image.as_posix())

    # tess_angle = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT).get('rotate', 0)
    tess_osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT, config="-c min_characters_to_try=50")
    conf = tess_osd['orientation_conf']
    tess_angle = tess_osd['rotate'] if conf >= 15 else 0
    
    if int(tess_angle) > 0:

        image = rotate_image(image, tess_angle)

    angle = only_text_angle(image)

    if angle > 0:
        image, _ = rotate_image_small(image, angle)

    return image


