import os

from pathlib import Path
import cv2 as cv
import pytesseract
from binarization import *
from convert.conversion import convert_from_path
from rotation import rotate_origin_image

RESULTS_PATH = Path('./tests/results')

if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir(exist_ok=True)

BINARIZTION_FUNCS = {
    'page_text_binarization_3': page_text_binarization_3
}


def binarize_image(image_path: str, binarization_func: str):
    image = cv.imread(Path(image_path).as_posix())
    bin_image = BINARIZTION_FUNCS[binarization_func](image)
    image_name = Path(image_path).stem
    cv.imwrite((RESULTS_PATH / f'{image_name}_binarized.jpg').as_posix(), bin_image)


def rotate_image(image_path: str):
    image = next(
        convert_from_path(Path(image_path), output_folder=RESULTS_PATH.as_posix(),
                              poppler_path='/tmp/envs/ocr/bin/', fmt='jpeg', output_file=Path(image_path).stem)
    )
    converted_pdf_path = RESULTS_PATH / f'{Path(image_path).stem}-1.jpg'
    converted_pdf_path = converted_pdf_path.rename(RESULTS_PATH / f'{Path(image_path).stem.split("-")[0]}.jpg')
    image_name = Path(image_path).stem
#     image = cv.imread(converted_pdf_path.as_posix())
    cv.imwrite(
        (RESULTS_PATH / f'{image_name}_rotated.jpg').as_posix(),
        rotate_origin_image(converted_pdf_path)
    )

def binarize_and_extract(image_path: str, binarization_func: str):
    next(
        convert_from_path(Path(image_path), output_folder=RESULTS_PATH.as_posix(),
                          poppler_path='/tmp/envs/ocr/bin/', fmt='jpeg', output_file=Path(image_path).stem)
    )
    converted_pdf_path = RESULTS_PATH / f'{Path(image_path).stem}-1.jpg'
    converted_pdf_path = converted_pdf_path.rename(RESULTS_PATH / f'{Path(image_path).stem.split("-")[0]}.jpg')
    image = cv.imread(converted_pdf_path.as_posix())
    bin_image = BINARIZTION_FUNCS[binarization_func](image)
  
    cv.imwrite((RESULTS_PATH / f'{converted_pdf_path.stem}_binarized.jpg').as_posix(), bin_image)
    image_hocr = pytesseract.image_to_pdf_or_hocr(
    bin_image,
    lang='rus',
    extension="hocr",
    config='--psm 6 --oem 3'
    )
    with open((RESULTS_PATH / f'{converted_pdf_path.stem}_extracted.hocr').as_posix(), 'wb') as hocr_file:
        hocr_file.write(image_hocr)

binarize_and_extract('./tests/resources/DOC001.pdf', binarization_func="page_text_binarization_3")
