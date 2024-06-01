import os
from kfp.components import create_component_from_func
from kfp import dsl
from minio import Minio
from minio.error import S3Error
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict

# Function to retrieve data from Minio
def retrieve_data_from_minio(object_name: str) -> None:
    endpoint = "http://localhost:9000"
    bucket_name = "dvc-data"
    file_path = "/data/init_dataset.csv"
    
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    
    if not access_key or not secret_key:
        raise ValueError("Minio credentials (access key and secret key) must be provided via environment variables.")
    
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    
    try:
        client.fget_object(bucket_name, object_name, file_path)
        print(f"Data retrieved from Minio and saved to {file_path}.")
    except S3Error as e:
        print(f"Failed to retrieve data from Minio: {e}")

retrieve_data_from_minio_component = create_component_from_func(
    retrieve_data_from_minio,
    base_image="python:3.9",
    packages_to_install=["minio"]
)

# Function to preprocess data
def preprocess_data(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[str, str]:
    df = pd.read_csv(data_path)
    
    columns_to_convert = ['CQI1', 'CQI2', 'CQI3', 'cSTD CQI', 'cMajority', 'c25 P', 'c50 P', 'c75 P', 'RSRP1', 'RSRP2', 
                          'RSRP3', 'pMajority', 'p25 P', 'p50 P', 'p75 P', 'RSRQ1', 'RSRQ2', 'RSRQ3', 'qMajority', 
                          'q25 P', 'q50 P', 'q75 P', 'SNR1', 'SNR2', 'SNR3', 'sMajority', 's25 P', 's50 P', 's75 P']
    
    df[columns_to_convert] = df[columns_to_convert].astype(float)
    df[columns_to_convert] = df[columns_to_convert].fillna(df[columns_to_convert].mean())
    df['Stall'].replace({'Yes': 1, 'No': 0}, inplace=True)
    
    X = df[columns_to_convert].values
    y = df['Stall'].values
    
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    training_path = "training_data.npz"
    testing_path = "testing_data.npz"
    
    np.savez_compressed(training_path, X_train=X_train.numpy(), y_train=y_train.numpy())
    np.savez_compressed(testing_path, X_test=X_test.numpy(), y_test=y_test.numpy())
    
    return training_path, testing_path

preprocess_data_component = create_component_from_func(
    preprocess_data,
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "torch", "scikit-learn", "imblearn"]
)

# Function to standardize and convert data
def standardize_and_convert(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = torch.from_numpy(X_train).type(torch.float32)
    X_test = torch.from_numpy(X_test).type(torch.float32)
    y_train = torch.from_numpy(y_train).type(torch.float32)
    y_test = torch.from_numpy(y_test).type(torch.float32)
    
    return X_train, X_test, y_train, y_test

standardize_and_convert_component = create_component_from_func(
    standardize_and_convert,
    base_image="python:3.9",
    packages_to_install=["numpy", "torch", "scikit-learn"]
)

# Function to train model
def train_model(model: torch.nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, 
                loss_fn: torch.nn.Module = torch.nn.BCEWithLogitsLoss(), lr: float = 0.0001, epochs: int = 3500, seed: int = 42, print_every: int = 500) -> Dict[str, float]:
    torch.manual_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc
    
    for epoch in range(epochs):
        model.train()
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        
        if epoch % print_every == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
    
    metrics = {
        'final_train_loss': loss.item(),
        'final_train_accuracy': acc,
        'final_test_loss': test_loss.item(),
        'final_test_accuracy': test_acc
    }
    
    return metrics

train_model_component = create_component_from_func(
    train_model,
    base_image="python:3.9",
    packages_to_install=["torch"]
)

# Pipeline to preprocess data
@dsl.pipeline(
    name="DataPreprocessingPipeline",
    description="A pipeline to preprocess data and split it into training and testing sets."
)
def data_preprocessing_pipeline(data_path: str):
    preprocess_data_task = preprocess_data_component(data_path=data_path)

pipeline_filename = 'data_preprocessing_pipeline.yaml'
kfp.compiler.Compiler().compile(pipeline=data_preprocessing_pipeline, package_path=pipeline_filename)

kfp_endpoint = "http://localhost:3000"
client = kfp.Client(host=kfp_endpoint)

experiment_name = "DataPreprocessingExperiment"
pipeline_name = "DataPreprocessingPipeline"

experiment = client.create_experiment(experiment_name)
run = client.create_run_from_pipeline_package(
    package_path=pipeline_filename,
    experiment_id=experiment.experiment_id,
    run_name="DataPreprocessingRun"
)

print(f"Pipeline run created with ID: {run.run_id}")

# Pipeline to retrieve data from Minio
@dsl.pipeline(
    name="RetrieveDataFromMinioPipeline",
    description="A pipeline to retrieve data from a Minio database."
)
def retrieve_data_pipeline(object_name: str):
    retrieve_data_task = retrieve_data_from_minio_component(object_name=object_name)

pipeline_filename = 'retrieve_data_pipeline.yaml'
kfp.compiler.Compiler().compile(pipeline=retrieve_data_pipeline, package_path=pipeline_filename)

experiment_name = "RetrieveDataFromMinioExperiment"
pipeline_name = "RetrieveDataFromMinioPipeline"
parameters = {"object_name": "init_dataset.csv"}

experiment = client.create_experiment(experiment_name)
run = client.create_run_from_pipeline_package(
    package_path=pipeline_filename,
    experiment_id=experiment.experiment_id,
    run_name="RetrieveDataFromMinioRun",
    pipeline_parameters=parameters,
)

print(f"Pipeline run created with ID: {run.run_id}")
