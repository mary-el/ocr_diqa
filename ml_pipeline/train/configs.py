from sklearn.svm import SVR

DS_PATH_SPLIT = 'data/smartdoc_ds_split.pkl'
MODEL = SVR
MODEL_PARAMS = {'kernel': 'rbf', 'gamma': 0.1, 'degree': 2}
MODEL_PATH = 'models/svr'
