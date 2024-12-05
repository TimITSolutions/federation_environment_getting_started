import mlflow
import os
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import os


"""
This example code demonstrates how to submit code to the evaluation platform fw.zgt.nl that will log against the 
provided mlflow instance. Follow the given steps.

STEP 1: 
    Create an account at fe.zgt.nl and login.

STEP 2:
    Copy the Mlflow Username and Password from the website into the below variables:

"""
username = "REPLACE_WITH_YOUR_USERNAME"
password = "REPLACE_WITH_YOUR_PASSWORD"
"""

STEP 3:
    Commit everything to the master branch:
        git commit -am "added my own credentials"


STEP 4:
    Export your code using the git archive feature:
        git archive --format=zip --output submission.zip master


STEP 5:
    Upload the resulting ZIP file to the platform.


Following these steps ensures that you will upload everything in the correct format.

"""

def check_directory(path):
    """
    This function is checks if the a given directory path exists.
    Used to test if the dataset is accessible 
    """
    if os.path.exists(path) and os.path.isdir(path):
        contents = os.listdir(path)
        if contents:
            print("Path exists and is not empty")
            for item in contents:
               print(f"- {item}")
            return True
        else:
            print("Path exists but empty")
            return False
    else:
        print("Path does not exist")
        return False


# set username and passwort through environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = password

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://localhost:3001")

# Create a new MLflow Experiment
mlflow.set_experiment(username)

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    if check_directory('/mnt/dataset/'):
        mlflow.log_metric("dataset-exists", 1)
    else:
        mlflow.log_metric("dataset-exists", 0)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    '''signature = infer_signature(X_train, lr.predict(X_train))

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    mlflow.log_image(image, key="dogs")

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )'''