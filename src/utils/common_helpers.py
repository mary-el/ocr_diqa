import cv2 as cv

import numpy as np

def whitening_lines(
    gray_image: np.ndarray, vertical_threshold=0.05, horizontal_threshold=0.05
):
    """
    Whiten vertical and horizontal lines on an image.
    Input:
        gray_image: array image
        vertical_threshold: threshold as a ratio of line length
            to image height, from 0 to 1
        horizontal_threshold: threshold as a ratio of line length
            to image width, from 0 to 1
    Return:
        grayscale image with lines longer then threshold beeing whiten
    """
    if len(gray_image.shape) > 2:
        gray_image = cv.cvtColor(gray_image, cv.COLOR_BGR2GRAY)
    h, w = gray_image.shape

    vertical_threshold = h * vertical_threshold
    horizontal_threshold = w * horizontal_threshold
    horizontal_rect_threshold = w * 0.4
    good_rect_area_threshold = 0.6

    image = cv.bilateralFilter(gray_image, 9, 75, 75)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    bw_img = cv.bitwise_not(image)

    # Find vertical lines
    image = cv.dilate(bw_img, np.ones((1, 5), np.uint8), iterations=1)
    image = cv.erode(image, np.ones((21, 1), np.uint8), iterations=1)
    image = cv.erode(image, np.ones((5, 5), np.uint8), iterations=1)
    image = cv.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
    image = cv.dilate(image, np.ones((21, 1), np.uint8), iterations=1)
    image = cv.erode(image, np.ones((3, 3), np.uint8), iterations=1)
    image = cv.dilate(image, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Remove too small vertical lines
    for i in contours:
        rectangle = cv.minAreaRect(i)
        box = cv.boxPoints(rectangle)
        box = np.int0(box)
        ys = box[:, 1]
        diff = max(ys) - min(ys)
        if diff < vertical_threshold:
            cv.fillConvexPoly(image, i, (0, 0, 0))

    vertical_mask = image == 255
    gray_image[vertical_mask] = 255

    # Find horizontal lines
    image = cv.dilate(bw_img, np.ones((5, 1), np.uint8), iterations=1)
    image = cv.erode(image, np.ones((1, 21), np.uint8), iterations=1)
    image = cv.erode(image, np.ones((5, 5), np.uint8), iterations=1)
    image = cv.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
    image = cv.dilate(image, np.ones((1, 21), np.uint8), iterations=1)
    image = cv.erode(image, np.ones((3, 3), np.uint8), iterations=1)
    image = cv.dilate(image, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Remove too small horizontal lines
    for i in contours:

        area = cv.contourArea(i)
        rect = cv.boundingRect(i)
        break_contour = (
            True
            if area < rect[2] * rect[3] * good_rect_area_threshold
            else False
        )

        rectangle = cv.minAreaRect(i)
        box = cv.boxPoints(rectangle)
        box = np.int0(box)
        xs = box[:, 0]
        diff = max(xs) - min(xs)
        if diff < horizontal_threshold or (
            break_contour and diff < horizontal_rect_threshold
        ):
            cv.fillConvexPoly(image, i, (0, 0, 0))

    horizontal_mask = image == 255

    black_pixel_count = len(np.where(horizontal_mask == 0)[0])
    white_pixel_count = len(np.where(horizontal_mask == 1)[0])
    if black_pixel_count > 0 and white_pixel_count / black_pixel_count < 0.15:
        gray_image[horizontal_mask] = 255

    return gray_image, vertical_mask, horizontal_mask
