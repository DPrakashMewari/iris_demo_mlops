from src.components.dataingestion import ingest_data
from src.components.datatransformation import transform_data
from src.components.modeltraining import train_model
import mlflow

def train_pipeline(data_path):
    data = ingest_data(data_path)
    transformed_data = transform_data(data)
    train_model(data,transformed_data)
