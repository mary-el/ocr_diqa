import pytesseract

def get_text(image):
    return pytesseract.image_to_string(image, lang='rus+eng')
