from kfp.dsl import component, Input, Output, Dataset
from typing import NamedTuple

# Component for data preparation
@component(base_image="python:3.11.9", packages_to_install=['pandas==2.0.3', 'numpy==1.25.2', 'torch==2.3.0', 'scikit-learn==1.2.2', 'imblearn'])
def data_preparation(
    dataset_artifact: Input[Dataset],
    X_train_artifact: Output[Dataset], 
    X_test_artifact: Output[Dataset],
    y_train_artifact: Output[Dataset],
    y_test_artifact: Output[Dataset],
    test_size: float = 0.2, 
    random_state: int = 42
    ):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    import torch
    import os

    # Load dataset from Dataset artifact
    df = pd.read_pickle(dataset_artifact.path)

    # Handle null values and replace specific characters
    #df = df.replace([' ', '-',np.nan], 0) # There are null values
    df = df.replace([' ', '-', np.nan], np.nan)

    # Selective columns for mean calculation
    columns_to_convert = [
        'CQI1', 'CQI2', 'CQI3', 'cSTD CQI', 'cMajority', 'c25 P', 'c50 P', 'c75 P', 
        'RSRP1', 'RSRP2', 'RSRP3', 'pMajority', 'p25 P', 'p50 P', 'p75 P', 
        'RSRQ1', 'RSRQ2', 'RSRQ3', 'qMajority', 'q25 P', 'q50 P', 'q75 P', 
        'SNR1', 'SNR2', 'SNR3', 'sMajority', 's25 P', 's50 P', 's75 P'
    ]
    df[columns_to_convert] = df[columns_to_convert].astype(float)

    # Replace np.nan with mean values for selective columns
    df[columns_to_convert] = df[columns_to_convert].fillna(df[columns_to_convert].mean())

    # Convert 'Stall' column to numerical values
    df['Stall'].replace({'Yes': 1, 'No': 0}, inplace=True)

    X = df[columns_to_convert].values
    y = df['Stall'].values

    # Apply SMOTE for balancing the dataset
    # oversample = SMOTE(random_state=random_state)
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train_path = "/tmp/X_train.pt"
    X_test_path = "/tmp/X_test.pt"
    y_train_path = "/tmp/y_train.pt"
    y_test_path = "/tmp/y_test.pt"
    torch.save(X_train, X_train_path)
    os.rename(X_train_path, X_train_artifact.path)

    torch.save(X_test, X_test_path)
    os.rename(X_test_path, X_test_artifact.path)

    torch.save(y_train, y_train_path)
    os.rename(y_train_path, y_train_artifact.path)

    torch.save(y_test, y_test_path)
    os.rename(y_test_path, y_test_artifact.path)