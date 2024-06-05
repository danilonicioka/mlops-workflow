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
PIPELINE_ID = "my-pipeline-id"
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

# Data preparation var
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define a KFP component factory function for data ingestion
@dsl.component(base_image="python:3.12.3",packages_to_install=['gitpython==3.1.43', 'dvc==3.51.1','dvc-s3==3.2.0', 'pandas==2.0.3', 'numpy==1.25.2', 'torch==2.3.0', 'scikit-learn==1.2.2', 'imblearn==0.10.1'])
def main(
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
) -> NamedTuple('outputs', data_ingestion_result=str, data_preparation_result=str, model_training_result=str):
    from git import Repo
    from subprocess import run, CalledProcessError
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    import torch
    from torch import nn
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

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

    def data_preparation(data_path, test_size=0.2, random_state=42):
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

        result = "data prepararation done"

        return (result, X_train, X_test, y_train, y_test)
    
    def model_training(X_train, X_test, y_train, y_test, lr = 0.0001, epochs = 3500, seed = 42, print_every = 500):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build model with non-linear activation function
        class InterruptionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_1 = nn.Linear(in_features=29, out_features=200)
                self.layer_2 = nn.Linear(in_features=200, out_features=100)
                self.layer_3 = nn.Linear(in_features=100, out_features=1)
                self.relu = nn.ReLU() # <- add in ReLU activation function
                # Can also put sigmoid in the model
                # This would mean you don't need to use it on the predictions
                # self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # Intersperse the ReLU activation function between layers
                return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

        model_3 = InterruptionModel().to(device)
        print(model_3)

        # Setup loss and optimizer
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model_3.parameters(), lr=lr)

        def accuracy_fn(y_true, y_pred):
            correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
            acc = (correct / len(y_pred)) * 100
            return acc

        # Fit the model
        torch.manual_seed(seed)

        # Assuming X_train, y_train, X_test, y_test are already defined and are tensors
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        for epoch in range(epochs):
            # 1. Forward pass
            #model_3.train()
            y_logits = model_3(X_train).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

            # 2. Calculate loss and accuracy
            loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
            acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            ### Testing
            model_3.eval()
            with torch.no_grad():
                # 1. Forward pass
                test_logits = model_3(X_test).squeeze()
                #print(test_logits.shape)
                test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels

                # 2. Calculate loss and accuracy
                test_loss = loss_fn(test_logits, y_test)
                test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

            # Print out what's happening
            if epoch % print_every == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

        # Evaluate the final model
        model_3.eval()
        with torch.no_grad():
            y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

        predictions = y_preds.cpu().numpy() # if using cuda, otherwise y_pred.numpy()
        true_labels = y_test.cpu().numpy()

        print("=== Confusion Matrix ===")
        print(confusion_matrix(true_labels, predictions))
        print('\n')

        print("=== Score ===")
        accuracy = accuracy_score(true_labels, predictions)
        print('Accuracy: %f' % accuracy)

        precision = precision_score(true_labels, predictions, average='weighted')
        print('Precision: %f' % precision)
        recall = recall_score(true_labels, predictions, average='weighted')
        print('Recall: %f' % recall)

        microf1 = f1_score(true_labels, predictions, average='micro')
        print('Micro F1 score: %f' % microf1)
        macrof1 = f1_score(true_labels, predictions, average='macro')
        print('Macro F1 score: %f' % macrof1)

        target_names = ['No-Stall', 'Stall']

        # Print precision-recall report
        print(classification_report(true_labels, predictions, target_names=target_names))

        model_training_result = "model training done"
        return model_training_result

    # Call the functions
    clone_result = clone_repository_with_token(repo_url, cloned_dir, branch_name, github_username, github_token)
    configure_result = configure_dvc_remote(cloned_dir, remote_name, remote_url, minio_url, access_key, secret_key)
    dvc_pull_result = perform_dvc_pull(cloned_dir, remote_name)

    # Output dataset file
        # Define the target CSV file path as dataset.csv in the DVC file directory
    dataset_path = os.path.join(cloned_dir, dvc_file_dir, dvc_file_name)
    data_preparation_result, X_train, X_test, y_train, y_test = data_preparation(dataset_path)
    model_training_result = model_training(X_train, X_test, y_train, y_test)
    outputs = NamedTuple('outputs', data_ingestion_result=str, data_preparation_result=str, model_training_result=str)
    return outputs(f"{clone_result}, {configure_result}, {dvc_pull_result}", data_preparation_result, model_training_result)

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
) -> NamedTuple('pipe_outputs', data_ingestion_result=str, data_preparation_result=str, model_training_result=str):
    main_task = main(
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
        dvc_file_name=dvc_file_name)
    data_ingestion_result = main_task.outputs['data_ingestion_result']
    data_preparation_result = main_task.outputs['data_preparation_result']
    model_training_result = main_task.outputs['model_training_result']
    pipe_outputs = NamedTuple('pipe_outputs', data_ingestion_result=str, data_preparation_result=str, model_training_result=str)
    return pipe_outputs(data_ingestion_result, data_preparation_result, model_training_result)

# Compile the pipeline
pipeline_filename = f"{PIPELINE_NAME}.yaml"
kfp.compiler.Compiler().compile(
    pipeline_func=my_pipeline,
    package_path=pipeline_filename)

# Submit the pipeline to the KFP cluster
client = kfp.Client(host=KFP_HOST)  # Use the configured KFP host

client.create_run_from_pipeline_func(my_pipeline, enable_caching=False,
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