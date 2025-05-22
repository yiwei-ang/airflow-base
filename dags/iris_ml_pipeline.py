from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import subprocess

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='iris_ml_pipeline',
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'iris'],
) as dag:

    def download_data():
        from sklearn.datasets import load_iris
        import pandas as pd
        df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
        df['target'] = load_iris().target
        os.makedirs('/tmp/ml/', exist_ok=True)
        df.to_csv('/tmp/ml/iris.csv', index=False)

    def train_and_evaluate():
        subprocess.run(['python', '/opt/airflow/utils/train_model.py'], check=True)

    download_task = PythonOperator(
        task_id='download_iris_data',
        python_callable=download_data,
    )

    train_task = PythonOperator(
        task_id='train_and_evaluate',
        python_callable=train_and_evaluate,
    )

    download_task >> train_task
