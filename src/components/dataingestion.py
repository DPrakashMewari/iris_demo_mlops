import pandas as pd
import mlflow

def ingest_data(data_path):
    data = pd.read_csv(data_path)

    mlflow.log_param('data_path', data_path)

    return data
