import pickle
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor

from src.data.smartdoc_dataset import create_smartdoc_ds
from src.settings import RAW_DATA_PATH
from src.utils import catchtime


def split_dataset(load_file: str, save_file: str) -> None:
    with open(str(load_file), 'rb') as f:
        df = pickle.load(f)
    df.drop(columns=['spatial.gradients_4', 'tess_acc'], inplace=True)
    arr = np.array(df)
    X, y = arr[:, :-1], arr[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with open(save_file, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)


def load_split_ds(load_file: str) -> Tuple:
    with open(str(load_file), 'rb') as f:
        return pickle.load(f)


def grid_search(X_train: np.array, y_train: np.array) -> Dict:
    X_train = StandardScaler().fit_transform(MinMaxScaler().fit_transform(X_train))
    y_train /= 100.
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Elastic': ElasticNet(),
        'SVR': SVR(),
        'Tree': DecisionTreeRegressor(),
        # 'Gradient Boosting': GradientBoostingRegressor(n_estimators=4),
        # 'Bagging': BaggingRegressor(n_jobs=4)
    }
    params = {
        'Linear': {
        },
        'Ridge': {
            'alpha': [0.1, 0.5, 1.0],
            'solver': ['auto', 'sparse_cg', 'sag']
        },
        'Elastic': {
            'alpha': [0.1, 0.5, 1.0],
            'l1_ratio': [0.25, 0.5, 0.75],
            'tol': [1e-6, 1e-4, 1e-3],
            'selection': ['cyclic', 'random']
        },
        'SVR': {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'degree': [2, 3],
            'gamma': ['scale', 'auto', 0.1, 0.25, 0.2, 0.3]
        },
        'Tree': {
            'max_depth': [6, 12, 16]
        },
        'Gradient Boosting': {
            'loss': ['absolute_error'],
            'learning_rate': [0.01, 1, 2],
            'n_estimators': [30, 50, 100],
            'criterion': ['friedman_mse'],
            'max_depth': [3, 5, 10],
            'n_iter_no_change': [5],
        },
        'Bagging': {
            'base_estimator': [Ridge(alpha=0.1, solver='sparse_cg', positive=False, ), LinearRegression()],
            '_n_estimators': [50, 100],
            '_max_features': [0.1, 0.5, 1.]
        }
    }
    best_params = {}

    for name, model in models.items():
        grid = GridSearchCV(model, params[name], cv=5, n_jobs=4, scoring='r2',
                            verbose=True).fit(X_train, y_train)
        best_params[name] = (grid.best_params_, grid.best_score_)
    return best_params


def train_model(model, X_train: np.array, y_train: np.array) -> Pipeline:
    pipe = Pipeline(steps=[
        ('minmax_scaler', MinMaxScaler()),
        ('standard_scaler', StandardScaler()),
        ('regressor', model),
    ])
    y_train /= 100.
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(pipe: Pipeline, X_test: np.array, y_test: np.array) -> Tuple[np.array, np.array, Dict]:
    y_test /= 100.
    y_pred = pipe.predict(X_test)
    metrics = {
        'R2': r2_score,
        'MAE': mean_absolute_error,
        'MSE': mean_squared_error,
        'PLCC': lambda x, y: pearsonr(x, y)[0],
        'SROCC': lambda x, y: spearmanr(x, y, axis=0)[0]
    }
    results = {}
    for name, metric in metrics.items():
        results[name] = metric(y_test, y_pred)
    dump(pipe, f'models/pipeline_r2={results["R2"]:.3}.joblib')
    return y_test, y_pred, results


def plot(y_test: np.array, y_pred: np.array) -> None:
    plt.hist(y_pred - y_test, bins=100)
    plt.show()


def feature_importance(pipe: Pipeline, X_test: np.array, y_test: np.array) -> None:
    results = permutation_importance(pipe, X_test, y_test, scoring='r2')
    importance = results.importances_mean
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))


if __name__ == '__main__':
    # create_smartdoc_ds(RAW_DATA_PATH, r'data/small_ds.pkl')
    # ds_file = r'data/ds.pkl'
    ds_split_file = r'data/ds_split.pkl'
    # split_dataset(ds_file, ds_split_file)
    X_train, X_test, y_train, y_test = load_split_ds(ds_split_file)
    model = SVR(kernel='rbf', gamma=0.25)
    # # model = LinearRegression()
    pipe = train_model(model, X_train, y_train)
    # pipe = load('models/pipeline_r2=0.826.joblib')
    with catchtime():
        y_test, y_pred, results = evaluate_model(pipe, X_test, y_test)
    print(results)
    # # plot(y_test, y_pred)
    # # print(grid_search(X_train, y_train))
    # # feature_importance(pipe, X_test, y_test)
