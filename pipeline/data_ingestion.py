from kfp.dsl import component, Output, Dataset
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
PIPELINE_NAME = "Clone_and_Pull_Pipeline"
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