import logging
import os

from airflow.decorators import dag
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from configs import DS_PATH, DS_PATH_SPLIT, MODEL_PATH, RESULTS_PATH

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


# Create DAG
@dag("diqa_train", start_date=days_ago(0), schedule="@once", catchup=False)
def taskflow():
    # Task 1
    preprocess = DockerOperator(
        task_id="preprocess_ds",
        container_name="task__preprocess_ds",
        image="data:latest",
        command=f"python preprocess.py --ds_path {DS_PATH} --save_file {DS_PATH_SPLIT}",
        **dockerops_kwargs,
    )

    # Task 2
    train = DockerOperator(
        task_id="train_model",
        container_name="task__train_model",
        image="train:latest",
        command=f"python train.py --ds_path {DS_PATH_SPLIT} --model_path {MODEL_PATH}",
        **dockerops_kwargs,
    )

    evaluate = DockerOperator(
        task_id="evaluate_model",
        container_name="task__evaluate_model",
        image="evaluate:latest",
        command=f"python evaluate.py --ds_path {DS_PATH_SPLIT} --model_path {MODEL_PATH} --results_file {RESULTS_PATH}",
        **dockerops_kwargs,
    )

    preprocess >> train >> evaluate


taskflow()
