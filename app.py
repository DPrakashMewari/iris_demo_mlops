from flask import Flask, render_template, request
from src.components.dataingestion import ingest_data
from src.components.datatransformation import transform_data
from src.components.modeltraining import train_model
from src.pipeline.predict_pipeline import predict_pipeline
from src.pipeline.train_pipeline import train_pipeline
import mlflow
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    model = joblib.load('models/trained_model.joblib')

    # Load the input data from the form
    input_data = pd.DataFrame(request.form, index=[0])

    # Perform data transformation
    transformed_data = transform_data(input_data)

    # Make predictions using the model
    predictions = predict_pipeline(transformed_data, model)

    # Render the predictions in the frontend
    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
