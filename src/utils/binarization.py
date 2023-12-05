import logging
from typing import Callable, Dict, List, Optional

import cv2 as cv
import imutils
try:
    import doxapy
except ImportError:
    logging.warning("Cannot use doxa binarization function because doxa library is not installed")

import numpy as np
from pylsd import lsd
from skimage.exposure import exposure
from skimage.filters import threshold_sauvola

from src.utils.binarization_helpers import smart_whitening_lines, __sauvola_binarization_lut, __doxa_binarization

__all__ = (
    'page_text_binarization_3',
)

def page_text_binarization_3(image: np.ndarray, scale: int = 1, k_small_holes: float = 0.002, **kwargs) -> np.ndarray:
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = cv.resize(
        src=image,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv.INTER_LANCZOS4,
    )

    image = cv.GaussianBlur(image, (1, 3), 0)
    image = __sauvola_binarization_lut(image)

    image = cv.GaussianBlur(image, (1, 3), 0)
    kernel = np.ones((3, 1), np.uint8)
    image = cv.morphologyEx(src=image, op=cv.MORPH_OPEN, kernel=kernel)
    image = cv.morphologyEx(src=image, op=cv.MORPH_CLOSE, kernel=kernel)
    image = cv.GaussianBlur(image, (1, 1), 0)
    
    image = smart_whitening_lines(image, h_restore=True)

    return image


def page_text_binarization_1(image: np.ndarray) -> np.ndarray:
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (1, 3), 0)


    # kernel = np.ones((3, 1), np.uint8)
    # image = cv.morphologyEx(src=image, op=cv.MORPH_OPEN, kernel=kernel)
    # image = cv.morphologyEx(src=image, op=cv.MORPH_CLOSE, kernel=kernel)
    # image = cv.Canny(image, 100, 300, 3)

    allContours = cv.findContours(image.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    allContours = imutils.grab_contours(allContours)

    # сортировка контуров области по уменьшению и сохранение топ-1
    allContours = sorted(allContours, key=cv.contourArea, reverse=True)[:1]

    # aппроксимация контура
    perimeter = cv.arcLength(allContours[0], True)
    ROIdimensions = cv.approxPolyDP(allContours[0], 0.02 * perimeter, True)

    # изменение массива координат
    # ROIdimensions = ROIdimensions.reshape(4, 2)

    # список удержания координат ROI
    # rect = np.zeros((4, 2), dtype="float32")
    #
    # # наименьшая сумма будет у верхнего левого угла,
    # # наибольшая — у нижнего правого угла
    # s = np.sum(ROIdimensions, axis=1)
    # rect[0] = ROIdimensions[np.argmin(s)]
    # rect[2] = ROIdimensions[np.argmax(s)]
    #
    # # верх-право будет с минимальной разницей
    # # низ-лево будет иметь максимальную разницу
    # diff = np.diff(ROIdimensions, axis=1)
    # rect[1] = ROIdimensions[np.argmin(diff)]
    # rect[3] = ROIdimensions[np.argmax(diff)]
    #
    # # верх-лево, верх-право, низ-право, низ-лево
    # (tl, tr, br, bl) = rect
    #
    # # вычислить ширину ROI
    # widthA = np.sqrt((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2)
    # widthB = np.sqrt((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2)
    # maxWidth = max(int(widthA), int(widthB))
    #
    # # вычислить высоту ROI
    # heightA = np.sqrt((tl[0] -   bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    # heightB = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    # maxHeight = max(int(heightA), int(heightB))
    # image = cv.cvtColor(image, image.COLOR_BGR2GRAY)

    # image = __sauvola_binarization_lut(image)
    # _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    from skimage.filters import threshold_local
    # увеличить контраст в случае с документом

    T = threshold_local(image, 9, offset=20, method="gaussian")
    image = (image > T).astype("uint8") * 255
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # image = cv.morphologyEx(src=image, op=cv.MORPH_OPEN, kernel=kernel)
    # image = cv.morphologyEx(src=image, op=cv.MORPH_CLOSE, kernel=kernel)

    # image = cv.GaussianBlur(image, (1, 5), 0)

    return image


def doxa_text_binarization_sauvola(image: np.ndarray, **kwargs) -> np.ndarray:
    binarizator = doxapy.Binarization(doxapy.Binarization.Algorithms.ISAUVOLA)
    binarized_image = __doxa_binarization(image=image, binarizator=binarizator)
    # binarized_image = smart_whitening_lines(binarized_image, h_restore=True)
    return binarized_image


BLUR = 21
CANNY_THRESH = (10, 200)
MASK_DILATE_ITER, MASK_ERODE_ITER = 10, 10
MASK_COLOR = (0.0, 0.0, 1.0)


def remove_background_from_image(img) -> None:
    """
	Removes the Background of an Image that falls within a certain shade of gray.

        Parameters:
            image (str): The path of the image to be processed
            output (str): The path and name of the file with which to save the final result
	"""

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, CANNY_THRESH[0], CANNY_THRESH[1])
    edges = cv.dilate(edges, None)
    edges = cv.erode(edges, None)

    contour_info = list()

    contours, note = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv.isContourConvex(c),
            cv.contourArea(c),
        ))

    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    max_contour = contour_info[0]

    mask = np.zeros(edges.shape)

    cv.fillConvexPoly(mask, max_contour[0], 255)

    mask = cv.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv.GaussianBlur(mask, (BLUR, BLUR), 0)

    mask_stack = np.dstack([mask] * 3)

    mask_stack = mask_stack.astype('float32') / 255.0

    img = img.astype('float32') / 255.0

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)

    masked = (masked * 255).astype('uint8')

    red, green, blue = cv.split(img)

    result = cv.merge((red, green, blue, mask.astype('float32') / 255.0))

    return result

def func(img):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bit = cv.bitwise_not(gray)
    bit = bit[50:bit.shape[0] -50, 50:bit.shape[1] - 50]
    amtImage = cv.adaptiveThreshold(bit, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 35, 15)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv.dilate(amtImage,kernel,iterations = 2)
    kernel = np.ones((25,25),np.uint8)
    erosion = cv.erode(dilation, kernel, iterations = 10)
    bit = cv.bitwise_not(erosion)
    contours, hierarchy = cv.findContours(bit,  cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if (contours != 0):
        c = max(contours, key = cv.contourArea)
        x,y,w,h = cv.boundingRect(c)
        print(x, y, w, h)
    final = img[max(0, (y - 50)):(y + h) + min(250, (img.shape[0] - (y + h)) * 10 // 11), max(0, (x - 50)):(x + w) + min(250, (img.shape[1] - (x + w)) * 10 // 11)]

    return final
