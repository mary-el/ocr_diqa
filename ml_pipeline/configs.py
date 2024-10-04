from sklearn.svm import SVR

RAW_DATA_PATH = 'data/Dataset SmartDoc-QA'
DS_LINK = "https://zenodo.org/records/5293201/files/Dataset%20SmartDoc-QA.zip?download=1"
ZIP_PATH = "data/Dataset SmartDoc-QA.zip"
DS_PATH = 'data/smartdoc_ds.csv'
DS_PATH_SPLIT = 'data/smartdoc_ds_split.pkl'
IMAGE_PATH = 'data/image.jpg'
EPS = 0.001
N_PROC = 4
SIZE = 1024

MODEL = SVR
MODEL_PARAMS = {'kernel': 'rbf', 'gamma': 0.1, 'degree': 2}
MODEL_PATH = 'models/svr'

RESULTS_PATH = 'results/svr.csv'
