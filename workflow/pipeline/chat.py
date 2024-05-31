import kfp
from kfp import dsl
from typing import NamedTuple
import torch

@dsl.component(base_image="python:3.12.3", packages_to_install=['pandas', 'numpy', 'torch', 'scikit-learn', 'imblearn'])
def data_preparation(
    dataset: str, 
    data_path: str = 'dataset.csv', 
    test_size: float = 0.2, 
    random_state: int = 42
    ) -> NamedTuple('Outputs', [('X_train', torch.Tensor), ('X_test', torch.Tensor), ('y_train', torch.Tensor), ('y_test', torch.Tensor)]):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    import torch

    # Save the file content locally
    with open(data_path, 'wb') as local_file:
        local_file.write(dataset)

    df = pd.read_csv(data_path)

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

    outputs = NamedTuple('Outputs', [('X_train', torch.Tensor), ('X_test', torch.Tensor), ('y_train', torch.Tensor), ('y_test', torch.Tensor)])
    return outputs(X_train, X_test, y_train, y_test)
