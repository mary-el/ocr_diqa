import pickle
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# from src.data.smartdoc_dataset import create_smartdoc_ds
# from src.settings import RAW_DATA_PATH
# from src.utils.common import catchtime


def split_dataset(load_file: str, save_file: str) -> None:
    with open(str(load_file), 'rb') as f:
        df = pickle.load(f)
    df.drop(columns=['tess_acc'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)
    arr = np.array(df)
    X, y = arr[:, :-1], arr[:, -1]
    y = y / 100.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with open(save_file, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)


def load_split_ds(load_file: str) -> Tuple:
    with open(str(load_file), 'rb') as f:
        return pickle.load(f)


def grid_search(X_train: np.array, y_train: np.array, random=False) -> Dict:
    X_train = StandardScaler().fit_transform(MinMaxScaler().fit_transform(X_train))
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
        if random:
            random_search = RandomizedSearchCV(model, params[name], scoring='r2', n_iter=100, cv=5, random_state=42,
                                               n_jobs=-1)
            random_search.fit(X_train, y_train)
            best_params[name] = (random_search.best_params_, random_search.best_score_)
            print(f'{name}: {random_search.best_params_} {random_search.best_score_}')
        else:
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
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe: Pipeline, X_test: np.array, y_test: np.array) -> Tuple[np.array, np.array, Dict]:
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


def feature_selection(model, X_train, y_train, sfs_figure, sfs_df_filename, k_features, forward):
    sfs = SFS(model,
              k_features=k_features,
              forward=forward,
              floating=True,
              scoring='neg_mean_squared_error',
              cv=4,
              n_jobs=-1
              )
    sfs = sfs.fit(X_train, y_train)
    sfs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    sfs_df.to_csv(sfs_df_filename)

    plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    plt.savefig(sfs_figure)
    return sfs
