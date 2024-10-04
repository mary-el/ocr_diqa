RAW_DATA_PATH = 'data/Dataset SmartDoc-QA'
DS_LINK = "https://zenodo.org/records/5293201/files/Dataset%20SmartDoc-QA.zip?download=1"

DS_PATH = '/opt/airflow/data/smartdoc_ds.csv'
DS_PATH_SPLIT = '/opt/airflow/data/smartdoc_ds_split__{{ ds }}.pkl'
MODEL_PATH = '/opt/airflow/models/svr__{{ ds }}'

RESULTS_PATH = '/opt/airflow/results/svr__{{ ds }}.csv'
IMAGE_PATH = '/opt/airflow/data/image.jpg'
