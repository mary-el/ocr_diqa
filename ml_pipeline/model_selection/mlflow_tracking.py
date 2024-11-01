import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from autofeat import AutoFeatRegressor
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from optuna.integration import MLflowCallback
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from evaluate.evaluate import evaluate_model
from train.train import train_model

TRACKING_URI = 'http://127.0.0.1:5000'

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

EXPERIMENT_NAME = "Document Image Quality Assessment"
IMG_FOLDER = 'imgs'

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    EXPERIMENT_ID = experiment.experiment_id


def autofeat_transform(X_train, y_train, X_test):
    transformations = ['1+', 'sqrt', 'sin', 'cos', '1/', 'exp', 'log', '2^']
    feat = AutoFeatRegressor(feateng_steps=2, max_gb=1, transformations=transformations, n_jobs=-1)
    X_train_tr = feat.fit_transform(X_train, y_train)
    X_test_tr = feat.transform(X_test)
    with mlflow.start_run():
        mlflow.sklearn.log_model(feat, artifact_path='AutoFeatRegressor')
    return X_train_tr, X_test_tr


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


def sfs(model, X_train, y_train, figure_file='models/sfs.png', df_file='models/sfs.csv', forward=True,
        k_features=20):
    sfs = feature_selection(model, X_train, y_train, figure_file, df_file, k_features, forward)

    with mlflow.start_run():
        mlflow.log_artifact(figure_file, artifact_path='SFS')
        mlflow.log_artifact(df_file, artifact_path='SFS')
    return sfs


def objective(trial: optuna.Trial, X_train, y_train) -> float:
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 5),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5),
        "loss_function": "MAE",
        "task_type": "CPU",
        "random_seed": 0,
        "iterations": 300,
        "verbose": False,
    }
    model = CatBoostRegressor(**param)

    kf = KFold(n_splits=3)

    metrics = defaultdict(list)
    for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        train_x = X_train[train_index]
        train_y = y_train[train_index]
        val_x = X_train[val_index]
        val_y = y_train[val_index]

        model.fit(train_x, train_y)

        _, _, results = evaluate_model(model, val_x, val_y)
        for metric, value in results.items():
            metrics[metric].append(value)
    mae = np.median(metrics['MAE'])
    return mae


def optuna_optimization(ds_file, study_name):
    with open(ds_file, 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    RUN_NAME = 'model_bayesian_search'

    with mlflow.start_run(run_name=RUN_NAME, experiment_id=EXPERIMENT_ID) as run:
        run_id = run.info.run_id

    mlflc = MLflowCallback(
        tracking_uri=TRACKING_URI,
        metric_name="MAE",
        create_experiment=False,
        mlflow_kwargs={'experiment_id': EXPERIMENT_ID, 'tags': {'mlflow.run_id': run_id}}
    )
    study = optuna.create_study(direction='minimize', study_name=study_name,
                                sampler=optuna.samplers.TPESampler(), load_if_exists=True)
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=12, callbacks=[mlflc])
    best_params = study.best_params

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best params: {best_params}")

    plot_contour = optuna.visualization.plot_contour(study, params=['learning_rate', 'depth'])
    mlflow.log_figure(plot_contour, 'contour.png')

    plot_edf = optuna.visualization.plot_edf(study)
    mlflow.log_figure(plot_edf, 'edf.png')

    plot_optimization_history = optuna.visualization.plot_optimization_history(study)
    mlflow.log_figure(plot_optimization_history, 'optimization_history.png')

    plot_parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study, params=['learning_rate', 'depth'])
    mlflow.log_figure(plot_parallel_coordinate, 'parallel_coordinate.png')

    plot_param_importances = optuna.visualization.plot_param_importances(study)
    mlflow.log_figure(plot_param_importances, 'param_importances.png')

    plot_slice = optuna.visualization.plot_slice(study, params=['learning_rate', 'depth'])
    mlflow.log_figure(plot_slice, 'slice.png')

    plot_rank = optuna.visualization.plot_rank(study, params=['learning_rate', 'depth'])
    mlflow.log_figure(plot_rank, 'rank.png')

    plot_timeline = optuna.visualization.plot_timeline(study)
    mlflow.log_figure(plot_timeline, 'timeline.png')


def load_model():
    client = mlflow.MlflowClient(tracking_uri=TRACKING_URI, registry_uri=TRACKING_URI)
    experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    experiment_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
    ).sort_values(by="start_time", ascending=False)
    run = client.get_run(experiment_runs['run_id'][0])
    uri = client.get_model_version_download_uri('diqa', 1)


def run_experiment(base_model, params, ds_file, run_name, training_info=''):
    with open(ds_file, 'rb') as f:
        df = pickle.load(f)
    df = df.drop(columns=['tess_acc'])
    df = df.fillna(0)
    arr = np.array(df)
    X, y = arr[:, :-1], arr[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dataset = mlflow.data.from_pandas(df, name=ds_file, targets='fine_acc')

    model = base_model(**params)
    pipe = train_model(model, X_train, y_train)
    y_test, y_pred, results = evaluate_model(pipe, X_test, y_test)
    print(results)

    with mlflow.start_run(run_name=run_name, experiment_id=EXPERIMENT_ID):
        mlflow.log_params(params)
        mlflow.log_metrics(results)
        mlflow.set_tag("Training Info", training_info)
        signature = infer_signature(X_train, pipe.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="diqa_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="diqa",
        )
        mlflow.log_input(dataset)
