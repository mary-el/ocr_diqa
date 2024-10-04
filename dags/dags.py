import logging
import os

from airflow.decorators import dag
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from configs import DS_PATH, DS_PATH_SPLIT, MODEL_PATH, RESULTS_PATH, RAW_DATA_PATH, DS_LINK, IMAGE_PATH

log = logging.getLogger(__name__)

project_dir = os.getenv('CURRENT_DIR')

dockerops_kwargs = {
    "mount_tmp_dir": False,
    "mounts": [
        Mount(
            source=f"{project_dir}/data",
            target="/opt/airflow/data",
            type="bind",
        ),
        Mount(
            source=f"{project_dir}/models",
            target="/opt/airflow/models",
            type="bind",
        ),
        Mount(
            source=f"{project_dir}/results",
            target="/opt/airflow/results",
            type="bind",
        ),
    ],
    "retries": 1,
    "api_version": "1.30",
    "docker_url": "tcp://docker-socket-proxy:2375",
    "network_mode": "bridge",
}

@dag("diqa_load_data", start_date=days_ago(0), schedule=None, catchup=False)
def taskflow_load_data():
    load_dataset = DockerOperator(
        task_id="load_data",
        container_name="task__load_data",
        image="data:latest",
        command=f"python data/data_load.py --ds_path {RAW_DATA_PATH} --save_to {DS_PATH}",
        # command=f"python predict/predict.py --model_path {MODEL_PATH} --image_path {IMAGE_PATH}",
        **dockerops_kwargs,
    )
    load_dataset


# Create DAG
@dag("diqa_train", start_date=days_ago(0), schedule=None, catchup=False)
def taskflow_train():
    # Task 1
    preprocess = DockerOperator(
        task_id="preprocess_ds",
        container_name="task__preprocess_ds",
        image="data:latest",
        command=f"python data/preprocess.py --ds_path {DS_PATH} --save_file {DS_PATH_SPLIT}",
        **dockerops_kwargs,
    )

    # Task 2
    train = DockerOperator(
        task_id="train_model",
        container_name="task__train_model",
        image="train:latest",
        command=f"python train/train.py --ds_path {DS_PATH_SPLIT} --model_path {MODEL_PATH}",
        **dockerops_kwargs,
    )

    evaluate = DockerOperator(
        task_id="evaluate_model",
        container_name="task__evaluate_model",
        image="evaluate:latest",
        command=f"python evaluate/evaluate.py --ds_path {DS_PATH_SPLIT} --model_path {MODEL_PATH} --results_file {RESULTS_PATH}",
        **dockerops_kwargs,
    )

    preprocess >> train >> evaluate


@dag("diqa_predict", start_date=days_ago(0), schedule=None, catchup=False)
def taskflow_predict():
    predict = DockerOperator(
        task_id="predict",
        container_name="task__predict",
        image="predict:latest",
        # command=f"python predict/predict.py --model_path /opt/airflow/models/svr --image_path /opt/airflow/data/image.jpg",
        command=f"python predict/predict.py --model_path {MODEL_PATH} --image_path {IMAGE_PATH}",
        **dockerops_kwargs,
    )
    predict


taskflow_load_data()
taskflow_train()
taskflow_predict()
