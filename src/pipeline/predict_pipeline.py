from src.components.dataingestion import ingest_data
from src.components.datatransformation import transform_data
import mlflow
import pandas as pd
import joblib

# def predict_pipeline(data_path):
#     data = ingest_data(data_path)
#     transformed_data = transform_data(data)

#     model = joblib.load('models/trained_model.joblib')

#     predictions = model.predict(transformed_data)
    
#     return predictions


def predict_pipeline(data, model):
    # transformed_data = transform_data(data)
    predictions = model.predict(data)

    return predictions
