import cv2
import numpy as np

from utils.dataset.common import resize


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


def preprocess_image(img: np.array, width: int = 1024, min_page_size=100, rotate=False) -> np.array:
    """
    Prepare image for processing
    """
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize(img, width)
    img_blured = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        img_blured,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    morph = thresh
    best_approx, max_rectangle = find_page(morph)
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


def find_page(thresh: np.array):
    """
    Try to find big rectangular object on an image
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, thresh.shape[0] // 10, True)
        if 4 <= len(approx) <= 12:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < thresh.shape[1] - 20 or h < thresh.shape[0] - 20:
                return approx, cv2.boundingRect(cnt)
    return None, (0, 0, 0, 0)
