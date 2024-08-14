import kfp
from kfp import components
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics, ClassificationMetrics, OutputPath
import os
from dotenv import load_dotenv

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
KFP_HOST = "http://ml-pipeline.kubeflow.svc.cluster.local:8888"  # KFP host URL

# Define DVC remote configuration variables
REMOTE_NAME = "minio_remote"
REMOTE_URL = "s3://dvc-data"
MINIO_URL = "http://minio-svc.minio:9000"
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
DVC_FILE_DIR = 'data/external'
DVC_FILE_NAME = 'dataset.csv'

# Model serve config vars
MODEL_NAME = "youtubegoes5g"
FRAMEWORK = "pytorch"
NAMESPACE = "kubeflow-user-example-com"
SVC_ACCOUNT = "pipeline-runner"

with open(os.environ['KF_PIPELINES_SA_TOKEN_PATH'], "r") as f:
    TOKEN = f.read()

# Define a KFP component factory function for data ingestion
@component(base_image="python:3.11.9",packages_to_install=['gitpython', 'dvc==3.51.1', 'dvc-s3==3.2.0', 'numpy==1.25.2', 'pandas==2.0.3'])
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
    dvc_file_name: str,
    dataset_artifact: Output[Dataset]
    ):
    from git import Repo
    from subprocess import run, CalledProcessError
    import os
    import pandas as pd

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

    # Save dataset with pandas in Dataset artifact
    pulled_dataset_path = os.path.join(cloned_dir, dvc_file_dir, dvc_file_name)
    tmp_dataset_path = "/tmp/" + dvc_file_name
    dataset = pd.read_csv(pulled_dataset_path)
    dataset.to_pickle(tmp_dataset_path)
    os.rename(tmp_dataset_path, dataset_artifact.path)
    
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

# Component for model training
@component(base_image="python:3.11.9", packages_to_install=['torch==2.3.0', 'scikit-learn==1.2.2', 'numpy==1.25.2'])
def model_training(
    X_train_artifact: Input[Dataset], 
    X_test_artifact: Input[Dataset],
    y_train_artifact: Input[Dataset],
    y_test_artifact: Input[Dataset],
    metrics: Output[Metrics], 
    classification_metrics: Output[ClassificationMetrics], 
    model_trained_artifact: Output[Model],
    model_trained_artifact_path: OutputPath(str),
    lr: float = 0.0001,
    epochs: int = 3500,
    print_every: int = 500
    ):
    import os
    import torch
    from torch import nn
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InterruptionModel().to(device)

    # Setup loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100
        return acc
    
    # Fit the model
    torch.manual_seed(42)
    epochs = epochs

    # Put all data on target device
    X_train = torch.load(X_train_artifact.path)
    X_test = torch.load(X_test_artifact.path)
    y_train = torch.load(y_train_artifact.path)
    y_test = torch.load(y_test_artifact.path)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for epoch in range(epochs):
        # 1. Forward pass
        y_logits = model(X_train).squeeze()

        y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy_fn(y_true=y_train,
                        y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model.eval()
        with torch.no_grad():
        # 1. Forward pass
            test_logits = model(X_test).squeeze()
            #print(test_logits.shape)
            test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
            # 2. Calcuate loss and accuracy
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                y_pred=test_pred)


        # Print out what's happening
        #if epoch % print_every == 0:
        #    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

        model.eval()
        with torch.no_grad():
            y_preds = torch.round(torch.sigmoid(model(X_test))).squeeze()

        if device == "cuda":
            predictions = y_preds.cpu().numpy() #if it is cuda, then this, otherwise y_pred.numpy()
            true_labels = y_test.cpu().numpy()
        else:
            predictions = y_preds.numpy()
            true_labels = y_test.numpy()
        
        # Confusion Matrix
        cmatrix = confusion_matrix(true_labels, predictions)
        #print("Confusion Matrix:", cmatrix)

        # Metrics
        accuracy = accuracy_score(true_labels, predictions)
        metrics.log_metric("Accuracy", accuracy)
        #print('Accuracy: %f' % accuracy)

        precision = precision_score(true_labels,  predictions, average='weighted')
        metrics.log_metric("Precision", precision)
        #print('Precision: %f' % precision)

        recall = recall_score(true_labels, predictions, average='weighted')
        metrics.log_metric("Recall", recall)
        #print('Recall: %f' % recall)

        microf1 = f1_score(true_labels, predictions, average='micro')
        metrics.log_metric("Micro F1 score", microf1)
        #print('Micro F1 score: %f' % microf1)

        macrof1 = f1_score(true_labels, predictions, average='macro')
        metrics.log_metric("Macro F1 score", macrof1)
        #print('Macro F1 score: %f' % macrof1)

        target_names = ['No-Stall', 'Stall']
        # Print precision-recall report
        #print(classification_report(true_labels, predictions, target_names=target_names))

        # Classification Metrics artifact
        cmatrix = cmatrix.tolist()
        target_names = ['No-Stall', 'Stall']
        classification_metrics.log_confusion_matrix(target_names, cmatrix)

        # Save model
        model_path = "/tmp/model.pt"
        torch.save(model.state_dict(), model_path)
        os.rename(model_path, model_trained_artifact.path)

        with open(model_trained_artifact_path, 'w') as f:
            f.write(model_trained_artifact.path)
        
@component(base_image="python:3.11.9", packages_to_install=['kserve==0.13.0','kubernetes==30.1.0'])
def model_serving(
    namespace: str,
    model_name: str
):
    # Create kserve instance
    from kubernetes import client 
    from kserve import KServeClient, constants, V1beta1InferenceService, V1beta1InferenceServiceSpec, V1beta1PredictorSpec, V1beta1TorchServeSpec
    from datetime import datetime
    import time
    from subprocess import run
    
    #get model uri
    uri = "pvc://model-store-claim"
    
    #TFServing wants this type of structure ./models/1/model
    # the number represent the model version
    # in this example we use only 1 version
    
    #Inference server config
    now = datetime.now()
    kserve_version='v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    isvc = V1beta1InferenceService(api_version=api_version,
                                   kind=constants.KSERVE_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=model_name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false'}),
                                   spec=V1beta1InferenceServiceSpec(
                                   predictor=V1beta1PredictorSpec(
                                       service_account_name="sa-minio-kserve",
                                       pytorch=(V1beta1TorchServeSpec(
                                           storage_uri=uri))))
    )

    KServe = KServeClient()
    
    #replace old inference service with a new one
    try:
        KServe.delete(name=model_name, namespace=namespace)
        print("Old model deleted")
    except:
        print("Couldn't delete old model")
    time.sleep(10)
    
    KServe.create(isvc)

@pipeline
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
    dvc_file_name: str,
    namespace: str,
    model_name: str
):
    data_ingestion_task = data_ingestion(
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
    dataset_artifact = data_ingestion_task.output
    data_preparation_task = data_preparation(dataset_artifact=dataset_artifact)
    X_train_artifact = data_preparation_task.outputs["X_train_artifact"]
    X_test_artifact = data_preparation_task.outputs["X_test_artifact"]
    y_train_artifact = data_preparation_task.outputs["y_train_artifact"]
    y_test_artifact = data_preparation_task.outputs["y_test_artifact"]
    model_training_task = model_training(X_train_artifact=X_train_artifact, 
                                         X_test_artifact=X_test_artifact, 
                                         y_train_artifact=y_train_artifact, 
                                         y_test_artifact=y_test_artifact)
    model_trained_artifact_path = model_training_task.outputs["model_trained_artifact_path"]
    model_serving_task = model_serving(namespace=namespace, 
                                       model_name=model_name)

# Compile the pipeline
pipeline_filename = f"{PIPELINE_NAME}.yaml"
kfp.compiler.Compiler().compile(
    pipeline_func=my_pipeline,
    package_path=pipeline_filename)

# Submit the pipeline to the KFP cluster
client = kfp.Client(host=KFP_HOST, # Use the configured KFP host
                    existing_token=TOKEN)  


client.create_run_from_pipeline_func(
    my_pipeline,
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
        'dvc_file_name': DVC_FILE_NAME,
        'namespace': NAMESPACE,
        'model_name': MODEL_NAME
    })

#upload to Kubeflow 
# client.upload_pipeline(pipeline_package_path=pipeline_filename,
#                        pipeline_name="mlops",
#                        namespace = "kubeflow")