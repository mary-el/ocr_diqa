import numpy as np
import cv2 as cv

import math

from scipy.stats import mode

from common_helpers import whitening_lines

def grab_contours(cnts):
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    else:
        raise Exception(
            (
                "Contours tuple must have length 2 or 3, "
                "otherwise OpenCV changed their cv2.findContours return "
                "signature yet again. Refer to OpenCV's documentation "
                "in that case"
            )
        )

    return cnts


def get_cnt_rect_info(cnt):
    rect = cv.minAreaRect(cnt)  # вписываем прямоугольник
    max_rect = cv.boundingRect(cnt)  # описываем прямоугольник
    box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
    box = np.int0(box)  # округление координат
    center = (int(rect[0][0]), int(rect[0][1]))
    min_rect_area = int(rect[1][0] * rect[1][1])  # вычисление площади
    max_rect_area = int(max_rect[2] * max_rect[3])  # вычисление max площади

    # вычисление координат двух векторов, являющихся сторонам прямоугольника
    edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
    # edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
    edge2 = np.int0((box[1][0] - box[2][0], box[1][1] - box[2][1]))

    # выясняем какой вектор больше
    usedEdge = edge1
    if cv.norm(edge2) > cv.norm(edge1):
        usedEdge = edge2
    width_rect = int(cv.norm(usedEdge))
    reference = (1, 0)  # горизонтальный вектор, задающий горизонт

    # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
    angle = (
        180
        / math.pi
        * math.asin(
            (reference[0] * usedEdge[0] + reference[1] * usedEdge[1]) / (cv.norm(reference) * cv.norm(usedEdge))
        )
    )
    angle = angle if angle >= 0 else 180 + angle
    angle = round(abs(angle), 2)

    return min_rect_area, max_rect_area, center, width_rect, angle


def find_angle_small(image: np.ndarray, k_area=0.5):
    # debug_img = image.copy()
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Image properties
    height, width = image.shape
    # Define kernels
    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_11 = np.ones((1, 11), np.uint8)
    # Preprocessing
    image = cv.erode(image, kernel_3, iterations=1)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    image = cv.GaussianBlur(image, (9, 9), 0)
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    image = cv.bitwise_not(image)
    image = cv.dilate(image, kernel_3, iterations=1)
    image = cv.erode(image, kernel_3, iterations=1)
    image = cv.bitwise_not(image)
    image = cv.erode(image, kernel_11, iterations=3)
    image = cv.bitwise_not(image)
    image = cv.erode(image, kernel_11, iterations=10)
    image = cv.dilate(image, kernel_11, iterations=5)
    # Find contours
    contours = cv.findContours(image=image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    # image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # Filter contours by length
    angles = []
    for contour in contours:
        min_rect_area, max_rect_area, _, width_rect, angle = get_cnt_rect_info(contour)
        if min_rect_area > 100 and min_rect_area > max_rect_area * k_area and width_rect > width * 0.25:
            angles.append(angle)

    angles = [a for a in angles if a > max(angles) / 2]
    angles_int = list(map(int, angles))
    if not angles_int:
        return 0

    mode_angle = mode(angles_int)[0][0]
    angles_final = [angles[num] for num, angle in enumerate(angles_int) if angle == mode_angle]

    return np.mean(angles_final)


def rotate_image_small(image: np.ndarray, angle=None):
    if angle is None:
        angle = find_angle_small(image)
        if angle == 0 or abs(angle - 90) > 7:
            return image, angle
    height, width = image.shape[:2]
    cx, cy = width // 2, height // 2
    mask = np.full((height, width), 255, dtype=np.uint8)
    M = cv.getRotationMatrix2D((cx, cy), angle - 90, 1)
    # Find new image size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    # Fix Rotation matrix
    M[0, 2] += (new_width / 2) - cx
    M[1, 2] += (new_height / 2) - cy
    # Rotate image and mask
    image = cv.warpAffine(image, M, (new_width, new_height))
    mask = cv.warpAffine(mask, M, (new_width, new_height))
    # Find contours of the old image
    contours = cv.findContours(
        image=mask, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE
    )
    contours = grab_contours(contours)
    # Paint old image frame
    cv.drawContours(
        image=image,
        contours=contours,
        contourIdx=-1,
        color=(255, 255, 255),
        thickness=1,
    )
    # Change color background after rotation
    image[mask == 0] = 255
    return image, angle


def remove_shadow(image: cv):
    """Removes shadows from image:"""
    bgr_channels = cv.split(image)
    no_shadow_norm_channels = []

    for i in bgr_channels:
        dilated_image = cv.dilate(i, np.ones((7, 7), np.uint8))
        blured_image = cv.medianBlur(dilated_image, 21)
        diff_image = 255 - cv.absdiff(i, blured_image)
        norm_image = cv.normalize(
            diff_image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv.NORM_MINMAX,
            dtype=cv.CV_8UC1,
        )
        no_shadow_norm_channels.append(norm_image)

    return cv.merge(no_shadow_norm_channels)


def only_text_angle(image: np.ndarray):
    img = image.copy()

    img = remove_shadow(img)
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # бинаризация
    # TODO проверить бинаризацию через саволу
    v = np.median(img)
    sigma = 0.35 if v < 220 else 0.15
    low = int(max(110, (1.0 - sigma) * v))
    img = cv.threshold(img, low, 255, cv.THRESH_BINARY)[1]

    img, _, _ = whitening_lines(img, 0.03, 0.07)
    # img = del_tresh_lines(img, 0.03, 0.05, 1)

    _, angle = rotate_image_small(img)

    return angle


def rotate_image(image: np.ndarray, angle: int):
    if angle == 0:
        return image
    elif angle == 270:
        image = np.rot90(image)
    elif angle == 180:
        image = np.flip(image)
    elif angle == 90:
        image = np.rot90(image, -1)
    else:
        pass
    return image
