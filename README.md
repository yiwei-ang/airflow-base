"# airflow-base" 
helm upgrade airflow apache-airflow/airflow --namespace airflow --values airflow-values.yaml
kubectl get pods -n airflow  