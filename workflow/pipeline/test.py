import os
from kfp.components import create_component_from_func
from kfp import dsl
from minio import Minio
from minio.error import S3Error

## Data ingestion
# Define the function to retrieve data from Minio using a parameter for the object name
def retrieve_data_from_minio(object_name: str) -> None:
    """
    Retrieve data from Minio database and save it to a predefined file path.

    Parameters:
    - object_name: The name of the object in the Minio bucket.
    
    Predefined settings:
    - The Minio endpoint is set to 'http://localhost:9000'.
    - The Minio bucket name is set to 'dvc-data'.
    - The data is saved to the predefined local file path '/data/init_dataset.csv'.
    """
    # Predefined settings
    endpoint = "http://localhost:9000"
    bucket_name = "dvc-data"
    file_path = "/data/init_dataset.csv"

    # Retrieve Minio credentials from environment variables
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")

    # Check if Minio credentials are provided
    if not access_key or not secret_key:
        raise ValueError("Minio credentials (access key and secret key) must be provided via environment variables.")

    # Initialize the Minio client
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False  # Set to True if using HTTPS
    )

    # Retrieve the object from the specified bucket and save it to the file path
    try:
        client.fget_object(bucket_name, object_name, file_path)
        print(f"Data retrieved from Minio and saved to {file_path}.")
    except S3Error as e:
        print(f"Failed to retrieve data from Minio: {e}")

# Create the KFP component from the function
retrieve_data_from_minio_component = create_component_from_func(
    retrieve_data_from_minio,
    base_image="python:3.9",
    packages_to_install=["minio"]
)

## Data preparation =================================================================================

from typing import Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from kfp.components import create_component_from_func

def preprocess_data(
    data_path: str = 'Stall-Windows - Stall-3s.csv',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[str, str]:
    """
    Load data, preprocess it, and split it into training and testing sets.

    Parameters:
    - data_path: Path to the CSV file containing the data. Default is 'Stall-Windows - Stall-3s.csv'.
    - test_size: Proportion of the data to use as the test set. Default is 0.2 (20%).
    - random_state: Seed for random number generator. Default is 42.

    Returns:
    - Tuple of file paths to the training data and testing data.
    """
    # Load the data
    df = pd.read_csv(data_path)

    # Select columns to convert to numeric types and handle missing values
    columns_to_convert = ['CQI1', 'CQI2', 'CQI3', 'cSTD CQI',
                          'cMajority', 'c25 P', 'c50 P', 'c75 P', 'RSRP1', 'RSRP2', 'RSRP3',
                          'pMajority', 'p25 P', 'p50 P', 'p75 P', 'RSRQ1', 'RSRQ2', 'RSRQ3',
                          'qMajority', 'q25 P', 'q50 P', 'q75 P', 'SNR1', 'SNR2', 'SNR3',
                          'sMajority', 's25 P', 's50 P', 's75 P']

    # Convert selected columns to float type
    df[columns_to_convert] = df[columns_to_convert].astype(float)

    # Replace null values with mean values of the respective columns
    df[columns_to_convert] = df[columns_to_convert].fillna(df[columns_to_convert].mean())

    # Replace 'Yes' and 'No' in 'Stall' column with 1 and 0
    df['Stall'].replace({'Yes': 1, 'No': 0}, inplace=True)

    # Extract features (X) and target (y)
    X = df[columns_to_convert].values
    y = df['Stall'].values

    # Resample the dataset to balance classes using SMOTE
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert X and y to PyTorch tensors
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Save the training and testing sets to files
    training_path = "training_data.npz"
    testing_path = "testing_data.npz"

    np.savez_compressed(training_path, X_train=X_train.numpy(), y_train=y_train.numpy())
    np.savez_compressed(testing_path, X_test=X_test.numpy(), y_test=y_test.numpy())

    # Return file paths of training and testing data
    return training_path, testing_path

# Create the KFP component from the function
preprocess_data_component = create_component_from_func(
    preprocess_data,
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "torch", "scikit-learn", "imblearn"]
)

# Save the component spec to a YAML file if needed
preprocess_data_component.save('preprocess_data_component.yaml')

import os
from typing import Tuple

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kfp.components import create_component_from_func


def split_and_standardize_data(
    X_path: str,
    y_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[str, str, str, str]:
    """
    Split the data into training and testing sets, standardize the features,
    and convert data into PyTorch tensors.

    Parameters:
    - X_path: Path to the CSV file containing features data.
    - y_path: Path to the CSV file containing target data.
    - test_size: Proportion of the data to use as the test set. Default is 0.2 (20%).
    - random_state: Seed for random number generator. Default is 42.

    Returns:
    - Tuple of file paths to the saved training and testing sets (X_train_path, X_test_path, y_train_path, y_test_path).
    """
    # Load data from files
    X = np.load(X_path)
    y = np.load(y_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.float32)
    X_test = torch.from_numpy(X_test).type(torch.float32)
    y_train = torch.from_numpy(y_train).type(torch.float32)
    y_test = torch.from_numpy(y_test).type(torch.float32)

    # Save training and testing data to files
    X_train_path = "X_train.pth"
    X_test_path = "X_test.pth"
    y_train_path = "y_train.pth"
    y_test_path = "y_test.pth"

    torch.save(X_train, X_train_path)
    torch.save(X_test, X_test_path)
    torch.save(y_train, y_train_path)
    torch.save(y_test, y_test_path)

    return X_train_path, X_test_path, y_train_path, y_test_path


# Create the KFP component from the function
split_and_standardize_data_component = create_component_from_func(
    split_and_standardize_data,
    base_image="python:3.9",
    packages_to_install=["torch", "numpy", "scikit-learn"],
    output_component_file="split_and_standardize_data_component.yaml"
)

## Data pre-processing =================================================================================
from typing import Tuple
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from kfp.components import create_component_from_func

def standardize_and_convert(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize the features and convert the data to PyTorch tensors.

    Parameters:
    - X_train: Training features as a numpy array.
    - X_test: Testing features as a numpy array.
    - y_train: Training targets as a numpy array.
    - y_test: Testing targets as a numpy array.

    Returns:
    - X_train: Standardized training features as a PyTorch tensor.
    - X_test: Standardized testing features as a PyTorch tensor.
    - y_train: Training targets as a PyTorch tensor.
    - y_test: Testing targets as a PyTorch tensor.
    """
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert X_train, X_test, y_train, and y_test to PyTorch tensors
    X_train = torch.from_numpy(X_train).type(torch.float32)
    X_test = torch.from_numpy(X_test).type(torch.float32)
    y_train = torch.from_numpy(y_train).type(torch.float32)
    y_test = torch.from_numpy(y_test).type(torch.float32)

    return X_train, X_test, y_train, y_test

# Create the KFP component from the function
standardize_and_convert_component = create_component_from_func(
    standardize_and_convert,
    base_image="python:3.9",
    packages_to_install=["numpy", "torch", "scikit-learn"],
    output_component_file="standardize_and_convert_component.yaml"
)

## Model training =================================================================================

import torch
from kfp.components import create_component_from_func
from typing import Tuple, Dict

def train_model(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    loss_fn: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
    lr: float = 0.0001,
    epochs: int = 3500,
    seed: int = 42,
    print_every: int = 500
) -> Dict[str, float]:
    """
    Train a model using the provided training data and optimizer settings.

    Parameters:
    - model: PyTorch model to be trained.
    - X_train: Training features as a PyTorch tensor.
    - y_train: Training targets as a PyTorch tensor.
    - X_test: Testing features as a PyTorch tensor.
    - y_test: Testing targets as a PyTorch tensor.
    - loss_fn: Loss function for training (default: torch.nn.BCEWithLogitsLoss()).
    - lr: Learning rate for the optimizer (default: 0.0001).
    - epochs: Number of epochs to train (default: 3500).
    - seed: Seed for reproducibility (default: 42).
    - print_every: Frequency of printing training progress (default: every 500 epochs).

    Returns:
    - Dictionary containing final training and testing metrics.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Move data to the target device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define the accuracy function
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        # Forward pass: calculate logits
        y_logits = model(X_train).squeeze()

        # Convert logits to predictions using sigmoid and rounding
        y_pred = torch.round(torch.sigmoid(y_logits))

        # Calculate the loss
        loss = loss_fn(y_logits, y_train)

        # Calculate the accuracy
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        loss.backward()

        # Perform an optimization step
        optimizer.step()

        # Model evaluation mode (to disable gradients for inference)
        model.eval()
        with torch.no_grad():
            # Forward pass on the test set
            test_logits = model(X_test).squeeze()
            
            # Convert logits to predictions using sigmoid and rounding
            test_pred = torch.round(torch.sigmoid(test_logits))

            # Calculate the test loss
            test_loss = loss_fn(test_logits, y_test)
            
            # Calculate the test accuracy
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # Print metrics every print_every epochs
        if epoch % print_every == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

    # Return final metrics as a dictionary
    metrics = {
        'final_train_loss': loss.item(),
        'final_train_accuracy': acc,
        'final_test_loss': test_loss.item(),
        'final_test_accuracy': test_acc
    }
    
    return metrics

# Create the KFP component from the function
train_model_component = create_component_from_func(
    train_model,
    base_image="python:3.9",
    packages_to_install=["torch"],
    output_component_file="train_model_component.yaml"
)

## Model packaging and deployment =================================================================================



# Define a pipeline using KFP DSL =================================================================================
@dsl.pipeline(
    name="DataPreprocessingPipeline",
    description="A pipeline to preprocess data and split it into training and testing sets."
)
def data_preprocessing_pipeline():
    # Create a task using the preprocessing data component
    preprocess_data_task = preprocess_data_component()

# Compile the pipeline to a YAML file
pipeline_filename = 'data_preprocessing_pipeline.yaml'
kfp.compiler.Compiler().compile(pipeline=data_preprocessing_pipeline, package_path=pipeline_filename)

# Specify the KFP endpoint (localhost:3000)
kfp_endpoint = "http://localhost:3000"

# Create the KFP client
client = kfp.Client(host=kfp_endpoint)

# Run the pipeline
experiment_name = "DataPreprocessingExperiment"
pipeline_name = "DataPreprocessingPipeline"

# Create an experiment if not already existing
experiment = client.get_experiment(experiment_name)
if experiment is None:
    experiment = client.create_experiment(experiment_name)

# Submit the pipeline run
run = client.create_run_from_pipeline_package(
    package_path=pipeline_filename,
    experiment_id=experiment.experiment_id,
    run_name="DataPreprocessingRun"
)

print(f"Pipeline run created with ID: {run.run_id}")

## Pipeline =================================================================================
# Define the pipeline using KFP DSL
@dsl.pipeline(
    name="RetrieveDataFromMinioPipeline",
    description="A pipeline to retrieve data from a Minio database."
)
def retrieve_data_pipeline(object_name: str):
    # Create a task using the component and pass the object name as a parameter
    retrieve_data_task = retrieve_data_from_minio_component(object_name=object_name)

# Compile the pipeline to a YAML file
pipeline_filename = 'retrieve_data_pipeline.yaml'
kfp.compiler.Compiler().compile(pipeline=retrieve_data_pipeline, package_path=pipeline_filename)

# Specify the KFP endpoint
# Update the endpoint URL to 'localhost:3000' as requested
kfp_endpoint = "http://localhost:3000"

# Create the KFP client
client = kfp.Client(host=kfp_endpoint)

# Run the pipeline
# Specify the experiment name and parameters
experiment_name = "RetrieveDataFromMinioExperiment"
pipeline_name = "RetrieveDataFromMinioPipeline"
parameters = {
    "object_name": "init_dataset.csv"  # Provide the object name as input
}

# Create an experiment if not already existing
experiment = client.get_experiment(experiment_name)
if experiment is None:
    experiment = client.create_experiment(experiment_name)

# Submit the pipeline run
run = client.create_run_from_pipeline_package(
    package_path=pipeline_filename,
    experiment_id=experiment.experiment_id,
    run_name="RetrieveDataFromMinioRun",
    pipeline_parameters=parameters,
)

print(f"Pipeline run created with ID: {run.run_id}")