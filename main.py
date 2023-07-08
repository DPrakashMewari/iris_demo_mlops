from src.components.dataingestion import ingest_data
from src.components.datatransformation import transform_data
from src.components.modeltraining import train_model
from src.pipeline.predict_pipeline import predict_pipeline
from src.pipeline.train_pipeline import train_pipeline
import mlflow

def main():
    mlflow.set_tracking_uri('iris_mlflow')
    mlflow.set_experiment('iris_experiment')

    # Data Ingestion
    data_path = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    data = ingest_data(data_path)
    data.to_csv("data/raw/iris.csv")


    # Data Transformation
    transformed_data = transform_data(data)

    # Model Training
    train_model(data,transformed_data)

    # Prediction Pipeline
    predictions = predict_pipeline(data_path)
    print("Predictions:", predictions)

    # Train Pipeline
    train_pipeline(data_path)

if __name__ == '__main__':
    main()
