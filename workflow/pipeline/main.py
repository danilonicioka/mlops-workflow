import kfp
from kfp import dsl
import os
from dotenv import load_dotenv
from typing import NamedTuple

# Load environment variables from .env file
load_dotenv()

# Get GitHub username and token from environment variables
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Define configuration variables
REPO_URL = "https://github.com/danilonicioka/mlops-workflow.git"
CLONED_DIR = "mlops-workflow"
BRANCH_NAME = "tests"
PIPELINE_ID = "0"
PIPELINE_NAME = "mlops"
KFP_HOST = "http://localhost:3000"  # KFP host URL

# Define DVC remote configuration variables
REMOTE_NAME = "minio_remote"
REMOTE_URL = "s3://dvc-data"
MINIO_URL = "http://minio-svc.minio:9000"
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
DVC_FILE_DIR = 'data/external'
DVC_FILE_NAME = 'dataset.csv'

# Define a KFP component factory function for data ingestion
@dsl.component(base_image="python:3.12.3",packages_to_install=['gitpython', 'dvc==3.51.1','dvc-s3==3.2.0'])
def data_ingestion(
    repo_url: str,
    cloned_dir: str,
    branch_name: str,
    github_username: str,
    github_token: str,
    remote_name: str,
    remote_url: str,
    minio_url: str,
    access_key: str,
    secret_key: str,
    dvc_file_dir: str,
    dvc_file_name: str
) -> NamedTuple('outputs', result=str, dataset=str):
    from git import Repo
    from subprocess import run, CalledProcessError
    import os

    def clone_repository_with_token(repo_url, cloned_dir, branch_name, github_username, github_token):
        """Clone a Git repository using a GitHub token in the URL and specifying the branch."""
        try:
            # Construct the URL with the GitHub username and token
            url_with_token = f"https://{github_username}:{github_token}@{repo_url.split('//')[1]}"
            
            # Clone the repository from the specified branch
            repo = Repo.clone_from(url_with_token, cloned_dir, branch=branch_name)
            return "Repository cloned successfully"
        except Exception as e:
            return f"Error occurred during repository cloning: {e}"

    def configure_dvc_remote(cloned_dir, remote_name, remote_url, minio_url, access_key, secret_key):
        """Configure the Minio bucket as the DVC remote repository using the `dvc remote` commands."""
        try:
            # Add the remote
            run(
                ['dvc', 'remote', 'add', '-d', remote_name, remote_url],
                cwd=cloned_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Configure the endpoint URL
            run(
                ['dvc', 'remote', 'modify', remote_name, 'endpointurl', minio_url],
                cwd=cloned_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Configure access key ID
            run(
                ['dvc', 'remote', 'modify', remote_name, 'access_key_id', access_key],
                cwd=cloned_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Configure secret access key
            run(
                ['dvc', 'remote', 'modify', remote_name, 'secret_access_key', secret_key],
                cwd=cloned_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            return f'Successfully configured Minio bucket as DVC remote repository: {remote_name}'
        except CalledProcessError as e:
            # Log and raise any errors
            return f'Failed to configure DVC remote: {e.stderr}'

    def perform_dvc_pull(cloned_dir, remote_name):
        """Perform a DVC pull to synchronize local data with the remote repository."""
        try:
            # Run the `dvc pull` command
            result = run(['dvc', 'pull', '-r', remote_name], cwd=cloned_dir, capture_output=True, text=True)
            
            # Check if the command executed successfully
            if result.returncode != 0:
                # Log and raise an error if the command failed
                error_message = f"dvc pull failed with error: {result.stderr}"
                raise Exception(error_message)
            
            # Log successful operation
            return "Successfully pulled data from remote DVC repository"
            
        except Exception as e:
            # Log and handle the error
            return f"Error occurred during dvc pull: {e}"

    # Call the functions
    clone_result = clone_repository_with_token(repo_url, cloned_dir, branch_name, github_username, github_token)
    configure_result = configure_dvc_remote(cloned_dir, remote_name, remote_url, minio_url, access_key, secret_key)
    dvc_pull_result = perform_dvc_pull(cloned_dir, remote_name)

    # Output dataset file
        # Define the target CSV file path as dataset.csv in the DVC file directory
    dataset_path = os.path.join(cloned_dir, dvc_file_dir, dvc_file_name)
    f = open(dataset_path, 'r')
    dataset = f.read()
    outputs = NamedTuple('outputs', result=str, dataset=str)
    return outputs(f"{clone_result}, {configure_result}, {dvc_pull_result}", dataset)
    
# Component for data preparation
@dsl.component(base_image="python:3.12.3", packages_to_install=['pandas', 'numpy', 'torch', 'scikit-learn', 'imblearn'])
def data_preparation(
    dataset: str, 
    data_path: str = 'dataset.csv', 
    test_size: float = 0.2, 
    random_state: int = 42
    ) -> NamedTuple('outputs', result=str, X_train=list, X_test=list, y_train=list, y_test=list):
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

    outputs = NamedTuple('outputs', result=str, X_train=list, X_test=list, y_train=list, y_test=list)
    result = "data prepararation done"

    return outputs(result, X_train, X_test, y_train, y_test)

#

@dsl.pipeline
def my_pipeline(
    repo_url: str,
    cloned_dir: str,
    branch_name: str,
    github_username: str,
    github_token: str,
    remote_name: str,
    remote_url: str,
    minio_url: str,
    access_key: str,
    secret_key: str,
    dvc_file_dir: str,
    dvc_file_name: str
) -> str:
    # data_ingestion_task = data_ingestion(
    #     repo_url=repo_url,
    #     cloned_dir=cloned_dir,
    #     branch_name=branch_name,
    #     github_username=github_username,
    #     github_token=github_token,
    #     remote_name=remote_name,
    #     remote_url=remote_url,
    #     minio_url=minio_url,
    #     access_key=access_key,
    #     secret_key=secret_key,
    #     dvc_file_dir=dvc_file_dir,
    #     dvc_file_name=dvc_file_name)
    # data_ingestion_result = data_ingestion_task.outputs['result']
    # data_ingestion_dataset = data_ingestion_task.outputs['dataset']
    data_preparation_task = data_preparation(dataset=data_ingestion(
        repo_url=repo_url,
        cloned_dir=cloned_dir,
        branch_name=branch_name,
        github_username=github_username,
        github_token=github_token,
        remote_name=remote_name,
        remote_url=remote_url,
        minio_url=minio_url,
        access_key=access_key,
        secret_key=secret_key,
        dvc_file_dir=dvc_file_dir,
        dvc_file_name=dvc_file_name).outputs['dataset'])
    data_preparation_result = data_preparation_task.outputs['result']
    X_train = data_preparation_task.outputs['X_train']
    X_test = data_preparation_task.outputs['X_test']
    y_train = data_preparation_task.outputs['y_train']
    y_test = data_preparation_task.outputs['y_test']
    # pipe_outputs = NamedTuple('pipe_outputs', data_ingestion_result=str, data_preparation_result=str)
    return data_preparation_result

# Compile the pipeline
pipeline_filename = f"{PIPELINE_NAME}.yaml"
kfp.compiler.Compiler().compile(
    pipeline_func=my_pipeline,
    package_path=pipeline_filename)

# Submit the pipeline to the KFP cluster
client = kfp.Client(host=KFP_HOST)  # Use the configured KFP host
client.create_run_from_pipeline_func(
    my_pipeline,
    enable_caching=False,
    arguments={
        'repo_url': REPO_URL,
        'cloned_dir': CLONED_DIR,
        'branch_name': BRANCH_NAME,
        'github_username': GITHUB_USERNAME,
        'github_token': GITHUB_TOKEN,
        'remote_name': REMOTE_NAME,
        'remote_url': REMOTE_URL,
        'minio_url': MINIO_URL,
        'access_key': ACCESS_KEY,
        'secret_key': SECRET_KEY,
        'dvc_file_dir': DVC_FILE_DIR,
        'dvc_file_name': DVC_FILE_NAME
    })
