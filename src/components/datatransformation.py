import mlflow
from sklearn.preprocessing import StandardScaler  

# def transform_data(data):
#     # Perform data transformation
    
#     X = data.drop('variety', axis=1)
#     # Calculating the standardizing parameters that are the mean and standard deviation of the X_train dataset.  
#     standard_scaler = StandardScaler()    
#     transformed_data = standard_scaler.fit_transform(X)  

#     # Logging data transformation parameters to MLflow
#     mlflow.log_param('transformation_param', 'Standard Scaler')

#     return transformed_data
def transform_data(data):
    # Perform data transformation
    
    # X = data.drop('variety', axis=1)
    # Calculating the standardizing parameters that are the mean and standard deviation of the X_train dataset.  
    standard_scaler = StandardScaler()    
    transformed_data = standard_scaler.fit(data)  

    # Logging data transformation parameters to MLflow
    mlflow.log_param('transformation_param', 'Standard Scaler')

    return transformed_data