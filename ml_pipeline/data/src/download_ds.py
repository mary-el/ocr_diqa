import urllib.request
import zipfile
from configs import ZIP_PATH, DS_LINK


def download_smartdoc_ds(data_path):
    urllib.request.urlretrieve(DS_LINK, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(data_path)
