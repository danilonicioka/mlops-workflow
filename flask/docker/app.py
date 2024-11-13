from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, get_flashed_messages
from git import Repo
import os
from minio import Minio
import requests
import logging
from subprocess import run, CalledProcessError
from dotenv import load_dotenv
import kfp
import pandas as pd

# Load environment variables from an `.env` file
load_dotenv()

# Flask application setup
app = Flask(__name__)

# **Set the secret key**:
app.secret_key = os.environ.get('FLASK_SECRET_KEY')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration variables - some works better with this format
config = {
#     "GITHUB_REPO_URL": os.environ.get('GITHUB_REPO_URL', 'https://github.com/danilonicioka/mlops-workflow.git'),
#     "GITHUB_CLONED_DIR": os.environ.get('GITHUB_CLONED_DIR', 'mlops-workflow'),
#     "MODEL_INIT_DATASET_URL": os.environ.get('MODEL_INIT_DATASET_URL', 'https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv'),
#     "DVC_FILE_DIR": os.environ.get('DVC_FILE_DIR', 'data/external'),
#     "DVC_FILE_NAME": os.environ.get('DVC_FILE_NAME', 'dataset.csv'),
#     "GITHUB_DVC_BRANCH": os.environ.get('GITHUB_DVC_BRANCH', 'dvc'),
#     "DVC_BUCKET_NAME": os.environ.get('DVC_BUCKET_NAME', 'dvc-data'),
#     "MINIO_URL": os.environ.get('MINIO_URL', 'localhost:9000'),
#     "MINIO_ACCESS_KEY": os.environ.get('MINIO_ACCESS_KEY'),
#     "MINIO_SECRET_KEY": os.environ.get('MINIO_SECRET_KEY'),
#     "DVC_REMOTE_DB": os.environ.get('DVC_REMOTE_DB', 'minio_remote'),
    "GITHUB_USERNAME": os.environ.get('GITHUB_USERNAME'),
    "GITHUB_TOKEN": os.environ.get('GITHUB_TOKEN'),
#     "MODEL_NAME": os.environ.get('MODEL_NAME', 'youtubegoes5g'),
#     "NAMESPACE": os.environ.get('NAMESPACE', 'kubeflow-user-example-com'),
#     "LR": float(os.environ.get('LR', 0.0001)),  # Learning rate, converted to float
#     "EPOCHS": int(os.environ.get('EPOCHS', 3500)),  # Number of epochs, converted to int
#     "PRINT_FREQUENCY": int(os.environ.get('PRINT_FREQUENCY', 500)),  # Print frequency, converted to int
#     "OBJECT_NAME": os.environ.get('OBJECT_NAME', 'model-files'),
#     "SVC_ACC": os.environ.get('SVC_ACC', 'sa-minio-kserve'),
#     "PIPELINE_ID": os.environ.get('PIPELINE_ID', '7451916e-eee8-4c14-ad5f-8dee5aa61e3b'),
#     "VERSION_ID": os.environ.get('VERSION_ID', '264564bb-0ada-4095-920f-ae3bb9d8ca2e'),
#     "KFP_HOST": os.environ.get('KFP_HOST', 'http://localhost:8080'),
#     "KFP_AUTH_TOKEN": os.environ.get('KFP_AUTH_TOKEN'),  # Token for Kubeflow Pipelines authentication
#     "DEX_USER": os.environ.get('DEX_USER'),
#     "DEX_PASS": os.environ.get('DEX_PASS'),
#     "SVC_ACC_KFP": os.environ.get('SVC_ACC', 'default-editor'),
}

# Kubeflow variables
#KUBEFLOW_PIPELINE_NAME = "mlops"
KUBEFLOW_HOST_URL = 'http://localhost:8080'
#KUBEFLOW_HOST_URL = "http://ml-pipeline.kubeflow:8888"
KUBEFLOW_TOKEN = os.environ.get('KUBEFLOW_TOKEN'),  # Token for Kubeflow Pipelines authentication
# with open(os.environ['KF_PIPELINES_SA_TOKEN_PATH'], "r") as f:
#     KUBEFLOW_TOKEN = f.read()
KUBEFLOW_PIPELINE_ID = '34c26637-2901-4bdd-938b-188f66b74d84'
KUBEFLOW_VERSION_ID = 'd82a1e7c-66fc-486a-8ad7-2d10ada93f9d'
KUBEFLOW_SVC_ACC = 'default-editor'
K8S_API_TOKEN = os.environ.get('K8S_API_TOKEN')

# Dex variables
DEX_USER = os.environ.get('DEX_USER')
DEX_PASS = os.environ.get('DEX_PASS')

# DVC variables
DVC_REMOTE_DB = "minio_remote"
DVC_REMOTE_DB_URL = "s3://dvc-data"
DVC_FILE_DIR = 'data/external'
DVC_FILE_NAME = 'dataset.csv'
DVC_BUCKET_NAME = 'dvc-data'
DVC_FILE_PATH_EXT = os.path.join(f"{DVC_FILE_DIR}/{DVC_FILE_NAME}.dvc")

# Github variables
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO_URL = "https://github.com/danilonicioka/mlops-workflow.git"
GITHUB_CLONED_DIR = "mlops-workflow"
GITHUB_DVC_BRANCH = "dvc"
GITHUB_GITIGNORE_PATH = os.path.join(DVC_FILE_DIR, '.gitignore')
GITHUB_COMMIT_MSG_INIT = 'Add .dvc and .gitignore files'
GITHUB_COMMIT_MSG_APPEND = 'Update .dvc file'
GITHUB_MAIN_BRANCH = "main"

# MinIO variables
MINIO_CLUSTER_URL = "minio-service.kubeflow:9000"
MINIO_LOCAL_URL = 'localhost:9000'
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_MODEL_BUCKET_NAME = "model-files"
MINIO_MODEL_OBJECT_NAME = "model-store/youtubegoes5g/model.pt"

# Triggers variables
TRIGGER_TYPE = '0'
QUANTITY_FACTOR = 0.1
PERFORMANCE_FACTOR = 0.05
# Temp dir and files to save accuracy for trigger 3
TEMP_DIR = "tmp"
TEMP_FILE_ACC_IN_LAST_RUN = "accuracy_in_last_run.txt"
LAST_ACC_OBJECT_NAME = "accuracy-score/last_acc.txt"
TEMP_FILE_N_SAMPLES_SINCE_LAST_RUN = "n_samples_since_last_run.txt"
TEMP_FILE_N_SAMPLES_IN_LAST_RUN = "n_samples_in_last_run.txt"

# Model variables
MODEL_LR = 0.0001
MODEL_EPOCHS = 3500
MODEL_PRINT_FREQUENCY_PER_N_EPOCHS = 500
MODEL_NAME = "youtubegoes5g"
MODEL_INIT_DATASET_URL = 'https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv'

# Kserve variables
#MODEL_FRAMEWORK = "pytorch"
KSERVE_NAMESPACE = "kubeflow-user-example-com"
KSERVE_SVC_ACC = "sa-minio-kserve"
#MODEL_URI = "pvc://model-store-claim"
#MODEL_URI = "minio-service.kubeflow:9000/model-files"
INFERENCE_URL = "http://localhost:8080/predictions/youtubegoes5g"

####### Class to access kubeflow from outside the cluster

import re
from urllib.parse import urlsplit, urlencode

import kfp
import requests
import urllib3


class KFPClientManager:
    """
    A class that creates `kfp.Client` instances with Dex authentication.
    """

    def __init__(
        self,
        api_url: str,
        dex_username: str,
        dex_password: str,
        dex_auth_type: str = "local",
        skip_tls_verify: bool = False,
    ):
        """
        Initialize the KfpClient

        :param api_url: the Kubeflow Pipelines API URL
        :param skip_tls_verify: if True, skip TLS verification
        :param dex_username: the Dex username
        :param dex_password: the Dex password
        :param dex_auth_type: the auth type to use if Dex has multiple enabled, one of: ['ldap', 'local']
        """
        self._api_url = api_url
        self._skip_tls_verify = skip_tls_verify
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type
        self._client = None

        # disable SSL verification, if requested
        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # ensure `dex_default_auth_type` is valid
        if self._dex_auth_type not in ["ldap", "local"]:
            raise ValueError(
                f"Invalid `dex_auth_type` '{self._dex_auth_type}', must be one of: ['ldap', 'local']"
            )

    def _get_session_cookies(self) -> str:
        """
        Get the session cookies by authenticating against Dex
        :return: a string of session cookies in the form "key1=value1; key2=value2"
        """

        # use a persistent session (for cookies)
        s = requests.Session()

        # GET the api_url, which should redirect to Dex
        resp = s.get(
            self._api_url, allow_redirects=True, verify=not self._skip_tls_verify
        )
        if resp.status_code == 200:
            pass
        elif resp.status_code == 403:
            # if we get 403, we might be at the oauth2-proxy sign-in page
            # the default path to start the sign-in flow is `/oauth2/start?rd=<url>`
            url_obj = urlsplit(resp.url)
            url_obj = url_obj._replace(
                path="/oauth2/start", query=urlencode({"rd": url_obj.path})
            )
            resp = s.get(
                url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify
            )
        else:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for GET against: {self._api_url}"
            )

        # if we were NOT redirected, then the endpoint is unsecured
        if len(resp.history) == 0:
            # no cookies are needed
            return ""

        # if we are at `../auth` path, we need to select an auth type
        url_obj = urlsplit(resp.url)
        if re.search(r"/auth$", url_obj.path):
            url_obj = url_obj._replace(
                path=re.sub(r"/auth$", f"/auth/{self._dex_auth_type}", url_obj.path)
            )

        # if we are at `../auth/xxxx/login` path, then we are at the login page
        if re.search(r"/auth/.*/login$", url_obj.path):
            dex_login_url = url_obj.geturl()
        else:
            # otherwise, we need to follow a redirect to the login page
            resp = s.get(
                url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for GET against: {url_obj.geturl()}"
                )
            dex_login_url = resp.url

        # attempt Dex login
        resp = s.post(
            dex_login_url,
            data={"login": self._dex_username, "password": self._dex_password},
            allow_redirects=True,
            verify=not self._skip_tls_verify,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for POST against: {dex_login_url}"
            )

        # if we were NOT redirected, then the login credentials were probably invalid
        if len(resp.history) == 0:
            raise RuntimeError(
                f"Login credentials are probably invalid - "
                f"No redirect after POST to: {dex_login_url}"
            )

        # if we are at `../approval` path, we need to approve the login
        url_obj = urlsplit(resp.url)
        if re.search(r"/approval$", url_obj.path):
            dex_approval_url = url_obj.geturl()

            # approve the login
            resp = s.post(
                dex_approval_url,
                data={"approval": "approve"},
                allow_redirects=True,
                verify=not self._skip_tls_verify,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for POST against: {url_obj.geturl()}"
                )

        return "; ".join([f"{c.name}={c.value}" for c in s.cookies])

    def _create_kfp_client(self) -> kfp.Client:
        try:
            session_cookies = self._get_session_cookies()
        except Exception as ex:
            raise RuntimeError(f"Failed to get Dex session cookies") from ex

        # monkey patch the kfp.Client to support disabling SSL verification
        # kfp only added support in v2: https://github.com/kubeflow/pipelines/pull/7174
        original_load_config = kfp.Client._load_config

        def patched_load_config(client_self, *args, **kwargs):
            config = original_load_config(client_self, *args, **kwargs)
            config.verify_ssl = not self._skip_tls_verify
            return config

        patched_kfp_client = kfp.Client
        patched_kfp_client._load_config = patched_load_config

        return patched_kfp_client(
            host=self._api_url,
            cookies=session_cookies,
        )

    def create_kfp_client(self) -> kfp.Client:
        """Get a newly authenticated Kubeflow Pipelines client."""
        return self._create_kfp_client()

##########################################################

# Helper functions

def handle_dvc_errors(func):
    """Decorator to handle DVC-related errors and return appropriate error responses."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log error and return error response
            logger.error(f'DVC operation failed: {e}')
            return jsonify({'error': str(e)}), 400
    return wrapper

# Helper functions to initialize and manage Minio and DVC
def setup_minio_client(minio_url, access_key, secret_key, bucket_name):
    """Create a Minio client and ensure the bucket exists."""
    # Initialize Minio client with just the base URL (without path)
    client = Minio(
        minio_url,  # Ensure minio_url does not include a path, only the base URL (e.g., http://localhost:9000)
        access_key=access_key,
        secret_key=secret_key,
        secure=False  # Minio is using HTTP on localhost:9000
    )
    
    # Create the bucket if it does not exist
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        logger.info(f"Successfully created bucket: {bucket_name}")
    else:
        logger.info(f"Bucket {bucket_name} already exists")

    return client

def initialize_dvc_and_repo(cloned_dir):
    """Initialize the Git repository and DVC."""
    # Initialize the DVC repository using subprocess
    run(['dvc', 'init'], cwd=cloned_dir, capture_output=True, text=True, check=True)
    
    # Load the Git repository
    repo = Repo(cloned_dir)
    
    logger.info(f"Successfully initialized DVC in {cloned_dir}")
    return repo

def clone_repository_with_token(repo_url, cloned_dir, branch_name, github_username, github_token):
    """Clone a Git repository using a GitHub token in the URL and specifying the branch."""
    # Construct the URL with the GitHub username and token
    url_with_token = f"https://{github_username}:{github_token}@{repo_url.split('//')[1]}"
    
    # Clone the repository from the specified branch (in this case, 'dvc')
    repo = Repo.clone_from(url_with_token, cloned_dir, branch=branch_name)
    logger.info(f"Successfully cloned repository from {url_with_token} to {cloned_dir} on branch {branch_name}")
    return repo

def download_file(file_url, local_file_path):
    """Download a file from a given URL and save it locally."""
    try:
        # Request the file content
        response = requests.get(file_url)
        response.raise_for_status()
        
        # Save the file content locally
        with open(local_file_path, 'wb') as local_file:
            local_file.write(response.content)
        logger.info(f"Successfully downloaded file from {file_url} to {local_file_path}")
    except requests.RequestException as e:
        # Log and raise any download errors
        logger.error(f"Failed to download file: {e}")
        raise

def add_file_to_dvc(cloned_dir, dvc_file_path):
    """Add a file to DVC using the `dvc add` command."""
    try:
        # Run the `dvc add` command
        run(
            ['dvc', 'add', dvc_file_path],
            cwd=cloned_dir,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f'Successfully added {dvc_file_path} to DVC')
    except CalledProcessError as e:
        # Log and raise any errors
        logger.error(f'Failed to add file to DVC: {e.stderr}')
        raise Exception(f'Failed to add file to DVC: {e.stderr}')

def commit_and_push_changes(repo, file_paths, commit_message, branch_name):
    """Commit changes to Git and push them to the 'dvc' branch in GitHub."""
    try:
        # Add specified file paths to the Git index
        repo.index.add(file_paths)
        
        # Commit changes with the specified message
        repo.index.commit(commit_message)
        logger.info(f'Successfully committed changes to Git for files: {file_paths}')
        
        # Push changes to the 'dvc' branch in GitHub
        origin = repo.remotes.origin
        origin.push(refspec=f'HEAD:refs/heads/{branch_name}')  # Push changes to the 'dvc' branch
        
        logger.info(f'Successfully pushed changes to GitHub on the {branch_name} branch')
    except Exception as e:
        # Log the error and raise an exception
        logger.error(f'Failed to commit changes to Git or push to GitHub: {e}')
        raise

def configure_dvc_remote(cloned_dir, remote_name, remote_url, minio_url, access_key, secret_key):
    """Configure the Minio bucket as the DVC remote repository using the `dvc remote` commands."""
    try:
        # Add the remote
        run(
            ['dvc', 'remote', 'add', remote_name, remote_url],
            cwd=cloned_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Configure the endpoint URL
        run(
            ['dvc', 'remote', 'modify', remote_name, 'endpointurl', f'http://{minio_url}'],
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
        
        logger.info(f'Successfully configured Minio bucket as DVC remote repository: {remote_name}')
    except CalledProcessError as e:
        # Log and raise any errors
        logger.error(f'Failed to configure DVC remote: {e.stderr}')
        raise Exception(f'Failed to configure DVC remote: {e.stderr}')

def push_data_to_dvc(cloned_dir, remote_name):
    """Push data to the remote DVC repository using the `dvc push` command."""
    try:
        # Run the `dvc push` command to push data to the specified remote
        run(
            ['dvc', 'push', '--remote', remote_name],
            cwd=cloned_dir,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info('Successfully pushed data to remote DVC repository')
    except CalledProcessError as e:
        # Log and raise any errors
        logger.error(f'dvc push failed: {e.stderr}')
        raise Exception(f'dvc push failed: {e.stderr}')

def perform_dvc_pull(cloned_dir):
    """Perform a DVC pull to synchronize local data with the remote repository."""
    try:
        # Run the `dvc pull` command
        result = run(['dvc', 'pull'], cwd=cloned_dir, capture_output=True, text=True)
        
        # Check if the command executed successfully
        if result.returncode != 0:
            # Log and raise an error if the command failed
            error_message = f"dvc pull failed with error: {result.stderr}"
            logger.error(error_message)
            raise Exception(error_message)
        
        # Log successful operation
        logger.info("Successfully pulled data from remote DVC repository")
        
    except Exception as e:
        # Log and handle the error
        logger.error(f"Error occurred during dvc pull: {e}")
        raise

def append_csv_data(source_csv_path, target_csv_path):
    """Append data from the source CSV file to the target CSV file.

    This function reads the source and target CSV files as pandas DataFrames, 
    appends the data from the source to the target, and saves the result back 
    to the target CSV file.

    :param source_csv_path: Path to the source CSV file
    :param target_csv_path: Path to the target CSV file
    """
    try:
        # Read the source CSV file
        source_df = pd.read_csv(source_csv_path)
        logger.info(f"Successfully read source CSV file: {source_csv_path}")

        # Read the target CSV file
        target_df = pd.read_csv(target_csv_path)
        logger.info(f"Successfully read target CSV file: {target_csv_path}")

        # Append the source data to the target DataFrame
        updated_df = pd.concat([target_df, source_df])
        logger.info(f"Successfully appended data from {source_csv_path} to {target_csv_path}")

        # Save the updated DataFrame back to the target CSV file
        updated_df.to_csv(target_csv_path, index=False)
        logger.info(f"Successfully saved updated CSV file: {target_csv_path}")

    except pd.errors.EmptyDataError:
        # Handle the case where the CSV file might be empty
        logger.error(f"Empty CSV file encountered: {source_csv_path} or {target_csv_path}")
        raise

    except Exception as e:
        # Log any other errors that occur during the operation
        logger.error(f"Failed to append data from {source_csv_path} to {target_csv_path}: {e}")
        raise

# Helper function to execute an existing pipeline on Kubeflow
def execute_pipeline_run(kfp_host, dex_user, dex_pass, namespace, job_name, params, pipeline_id, version_id, svc_acc):
    # initialize a KFPClientManager
    kfp_client_manager = KFPClientManager(
        api_url=f'{kfp_host}/pipeline',
        skip_tls_verify=True,

        dex_username=dex_user,
        dex_password=dex_pass,

        # can be 'ldap' or 'local' depending on your Dex configuration
        dex_auth_type="local",
    )

    # get a newly authenticated KFP client
    # TIP: long-lived sessions might need to get a new client when their session expires
    client = kfp_client_manager.create_kfp_client()

    """Execute an existing pipeline on Kubeflow."""
    try:
        # Execute the pipeline
        run = client.run_pipeline(
            experiment_id="23d52751-4aeb-4e71-a47e-01c1ced25793",
            job_name=job_name,  # A name for the pipeline run
            params=params,
            pipeline_id=pipeline_id,
            version_id=version_id,
            enable_caching=False,
            service_account=svc_acc
        )

        logger.info(f"Pipeline run created successfully: {run.run_id}")
        return run
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        raise

# functions to save quantity data on temp files
def get_number_samples(file_path):
    """
    Function to get the number of samples (rows) in a CSV dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        int: Number of rows (samples) in the dataset.
    """
    try:
        df = pd.read_csv(file_path)
        return df.shape[0]
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def save_int_to_tempfile(int_value, dir_name, file_name):
    """
    Saves a int value to a specified directory and file name.

    Args:
        int_value (int): The int value to save.
        dir_name (str): The name of the directory to save the file in.
        file_name (str): The name of the file.
    
    Returns:
        str: The path to the file.
    """
    # Ensure the directory exists
    os.makedirs(dir_name, exist_ok=True)
    temp_file_path = os.path.join(dir_name, file_name)
    
    with open(temp_file_path, 'w') as temp_file:
        # Convert the int to a string, then write to file
        temp_file.write(str(int_value))
    
    return temp_file_path

def read_int_from_tempfile(temp_file_path):
    """
    Reads a int value from a specified file. Prints errors and returns 0.0 if the file does not exist or is blank.

    Args:
        temp_file_path (str): The path to the file.

    Returns:
        int: The int value read from the file, or 0.0 if the file does not exist, is blank, or an error occurs.
    """
    try:
        if not os.path.exists(temp_file_path):
            print(f"Error: File does not exist: {temp_file_path}")
            return 0.0
        
        with open(temp_file_path, 'r') as temp_file:
            content = temp_file.read().strip()  # Read and strip any extra whitespace

        if content == '':
            print(f"Error: File is blank: {temp_file_path}")
            return 0.0
        
        return int(content)  # Convert the read content to a int
    
    except (ValueError, IOError) as e:
        print(f"Error: {e}")
        return 0.0  # Return 0.0 if there is an error in reading or conversion

# def get_number_samples_since_last_run():
#     n_samples_since_last_run = read_int_from_tempfile(os.path.join(TEMP_DIR, TEMP_FILE_N_SAMPLES_SINCE_LAST_RUN))
#     return n_samples_since_last_run

# def get_number_samples_in_last_run():
#     n_samples_in_last_run = read_int_from_tempfile(os.path.join(TEMP_DIR, TEMP_FILE_N_SAMPLES_IN_LAST_RUN))
#     return n_samples_in_last_run

# get_number_samples_since_last_run and get_number_samples_in_last_run in one function
def get_number_samples_from_file(dir, file):
    n_samples = read_int_from_tempfile(os.path.join(dir, file))
    return n_samples

# def reset_number_samples_since_last_run():
#     save_int_to_tempfile(0.0, TEMP_DIR, TEMP_FILE_N_SAMPLES_SINCE_LAST_RUN)

# def increment_number_samples_since_last_run(new_quantity):
#     save_int_to_tempfile(new_quantity, TEMP_DIR, TEMP_FILE_N_SAMPLES_SINCE_LAST_RUN)

# reset and increment in one funtion update
# def update_number_samples_since_last_run(new_quantity):
#     save_int_to_tempfile(new_quantity, TEMP_DIR, TEMP_FILE_N_SAMPLES_SINCE_LAST_RUN)

# def update_number_samples_in_last_run(new_quantity):
#     save_int_to_tempfile(new_quantity, TEMP_DIR, TEMP_FILE_N_SAMPLES_IN_LAST_RUN)

# update_number_samples_since_last_run and update_number_samples_in_last_run in one function
def update_number_samples(new_quantity, dir, file):
    save_int_to_tempfile(new_quantity, dir, file)

# Routes

@app.route('/')
def home():
    # Check for flashed messages
    """Route to render the home page with buttons to navigate to /init and /append_csv routes."""
    success_message = get_flashed_messages(category_filter=["success"])
    fail_message = get_flashed_messages(category_filter=["fail"])

    return render_template('home.html', success_message=success_message, fail_message=fail_message)

@app.route('/init', methods=['GET'])
def init():
    """Initialize the application by cloning the repository, downloading file, and setting up DVC and Git."""
    try:
        # Define the target CSV file path as dataset.csv in the DVC file directory
        target_csv_path = os.path.join(GITHUB_CLONED_DIR, DVC_FILE_DIR, DVC_FILE_NAME)

        # Clone the repository from the dvc branch
        repo = clone_repository_with_token(
            GITHUB_REPO_URL, GITHUB_CLONED_DIR, GITHUB_DVC_BRANCH,
            config['GITHUB_USERNAME'], config['GITHUB_TOKEN']
        )

        # Download the file and save it as the target CSV file
        download_file(MODEL_INIT_DATASET_URL, target_csv_path)

        # Save the dataset's size for trigger type 2 (quantity type)
        new_quantity = get_number_samples(target_csv_path)
        # save number samples in last run to access later if needed
        update_number_samples(new_quantity, TEMP_DIR, TEMP_FILE_N_SAMPLES_IN_LAST_RUN)

        # Initialize DVC and Git repositories
        repo = initialize_dvc_and_repo(GITHUB_CLONED_DIR)

        # Specify the relative path to the target CSV file
        relative_target_csv_path = os.path.join(DVC_FILE_DIR, DVC_FILE_NAME)
        
        # Add the file to DVC using the relative path
        add_file_to_dvc(GITHUB_CLONED_DIR, relative_target_csv_path)
        
        # Commit changes to Git and push them to the 'dvc' branch in GitHub using relative paths
        commit_and_push_changes(repo, [DVC_FILE_PATH_EXT, GITHUB_GITIGNORE_PATH], GITHUB_COMMIT_MSG_INIT, GITHUB_DVC_BRANCH)

        # Set up Minio client and create a bucket if needed
        client = setup_minio_client(MINIO_LOCAL_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, DVC_BUCKET_NAME)

        # Configure Minio as the remote DVC repository
        remote_url = f's3://{DVC_BUCKET_NAME}'
        configure_dvc_remote(GITHUB_CLONED_DIR, DVC_REMOTE_DB, remote_url, MINIO_LOCAL_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)

        # Push data to remote DVC repository
        push_data_to_dvc(GITHUB_CLONED_DIR, DVC_REMOTE_DB)

        # Flash a success message and redirect to the home page
        flash('Successfully initialized the app, downloaded file as the target CSV file, added data to DVC, committed changes to GitHub, and pushed data to remote DVC repository.', 'success')
        return redirect(url_for('home'))

    except Exception as e:
        # Handle and log exceptions
        logger.error(f'Failed to initialize the application: {e}')

        # Flash the exception message
        flash(f'Error: {str(e)}', 'fail')
        
        # Redirect to home page to show the message
        return redirect(url_for('home'))

@app.route('/append_csv', methods=['GET', 'POST'])
@handle_dvc_errors
def append_csv():
    """Append data from the source CSV file to the target CSV file named as DVC_FILE_NAME, then update DVC and Git."""
    if request.method == 'GET':
        # Render the form using the submit_file.html template
        return render_template('submit_file.html')
    
    elif request.method == 'POST':
        # Handle file upload and append data
        if 'source_csv' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        # Get the uploaded file
        uploaded_file = request.files['source_csv']
        
        # Check if the file has an allowed extension (e.g., CSV)
        if not uploaded_file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400
        
        # Retrieve the trigger type from the form
        trigger_type = request.form.get('trigger_type')
        if not trigger_type:
            trigger_type = TRIGGER_TYPE
            #return jsonify({'error': 'Trigger type not provided'}), 400

        # Retrieve and convert quantity factor to float
        quantity_factor = request.form.get('quantity_factor', 10)
        if not quantity_factor:
            quantity_factor = QUANTITY_FACTOR
            #return jsonify({'error': 'Quantity factor not provided'}), 400

        try:
            quantity_factor = float(quantity_factor)/100.0  # Convert to float and percentage
        except ValueError:
            # Flash the error message
            flash(f'Error: Invalid quantity factor. Must be a number.', 'fail')
            
            # Redirect to home page to show the message
            return redirect(url_for('home'))
        
        # Retrieve and convert quantity factor to float
        performance_factor = request.form.get('performance_factor', 5)
        if not performance_factor:
            performance_factor = PERFORMANCE_FACTOR
            #return jsonify({'error': 'Performance factor not provided'}), 400

        try:
            performance_factor = float(performance_factor)/100.0  # Convert to float
        except ValueError:
            # Flash the exception message
            flash(f'Error:Invalid performance factor. Must be a number.', 'fail')
            
            # Redirect to home page to show the message
            return redirect(url_for('home'))

        logger.info(f"Trigger type received: {trigger_type}, Quantity factor: {quantity_factor}, Performance factor: {performance_factor}")
        
        # Define the path where the uploaded file will be saved
        source_csv_path = os.path.join(GITHUB_CLONED_DIR, 'temp_source.csv')
        
        # Save the uploaded file to a temporary location
        uploaded_file.save(source_csv_path)
        logger.info(f"Uploaded CSV file saved to {source_csv_path}")
        
        # Define the target CSV file path as DVC_FILE_NAME in the DVC file directory
        target_csv_path = os.path.join(GITHUB_CLONED_DIR, DVC_FILE_DIR, DVC_FILE_NAME)
        
        # Perform a DVC pull to ensure local data is up-to-date with the remote repository
        perform_dvc_pull(GITHUB_CLONED_DIR)

        # Get size of the new data and add the size of the accumulated previous new data (if the quantity factor condition was not achieved)
        new_quantity = get_number_samples(source_csv_path) + get_number_samples_from_file(TEMP_DIR, TEMP_FILE_N_SAMPLES_SINCE_LAST_RUN)

        # Get the size of the dataset when the last run was done, this way, we can compare the dataset before new data came
        previous_quantity = get_number_samples_from_file(TEMP_DIR, TEMP_FILE_N_SAMPLES_IN_LAST_RUN)

        # Append data from the source CSV file to the target CSV file
        append_csv_data(source_csv_path, target_csv_path)
        
        # Open the existing Git repository
        repo = Repo(GITHUB_CLONED_DIR)

        # Specify the relative path to DVC_FILE_NAME
        relative_target_csv_path = os.path.join(DVC_FILE_DIR, DVC_FILE_NAME)
        
        # Add the appended file to DVC using the relative path
        add_file_to_dvc(GITHUB_CLONED_DIR, relative_target_csv_path)
        
        # Push changes to the remote DVC repository
        push_data_to_dvc(GITHUB_CLONED_DIR, DVC_REMOTE_DB)
        
        # Commit changes to Git and push to GitHub for the updated .dvc file
        commit_and_push_changes(repo, [DVC_FILE_PATH_EXT], GITHUB_COMMIT_MSG_APPEND, GITHUB_DVC_BRANCH)

        # default is trigger_type == '0', not retraining
        exec_pipe = False
        flash_msg = "Successfully appended data from the uploaded CSV file to the target CSV file, added the file to DVC, pushed changes to the remote repository"
        job_name = "Not retraining trigger job"

        # Determine the job name based on the trigger type
        if trigger_type != '0': # if it is 0, skip this
            if trigger_type == '1':
                job_name = "Always retraining trigger job"
                flash_msg = f'{flash_msg}, and triggered pipeline run.'
                exec_pipe = True
            elif trigger_type == '2':
                job_name = "Quantity trigger job"
                if previous_quantity * quantity_factor < new_quantity:
                    flash_msg = f'{flash_msg}, and triggered pipeline run.'
                    exec_pipe = True
                    # reset the value of number of samples since last run and update the value of samples in last run because a new run was triggered
                    reset_value = 0
                    update_number_samples(reset_value, TEMP_DIR, TEMP_FILE_N_SAMPLES_SINCE_LAST_RUN)
                    new_dataset_size = previous_quantity + new_quantity
                    update_number_samples(new_dataset_size, TEMP_DIR, TEMP_FILE_N_SAMPLES_IN_LAST_RUN)
                else:
                    flash_msg = f'{flash_msg}. Pipeline not triggered, quantity factor condition not met'
                    # pipeline not triggered, so the new data need to be saved with the previous accumulated value
                    update_number_samples(new_quantity, TEMP_DIR, TEMP_FILE_N_SAMPLES_SINCE_LAST_RUN)
            elif trigger_type == '3':
                job_name = "Performance trigger job"
                flash_msg = f'{flash_msg}, and triggered pipeline run. The performance factor will be measured in the pipeline.'
                exec_pipe = True
            else:
                flash_msg = f'Trigger type {trigger_type} not defined'
                job_name = flash_msg

        # Parameters for the pipeline execution
        pipeline_params = {
            'github_repo_url': GITHUB_REPO_URL,
            'github_cloned_dir': GITHUB_CLONED_DIR,
            'github_dvc_branch': GITHUB_DVC_BRANCH,
            'github_username': config['GITHUB_USERNAME'], 
            'github_token': config['GITHUB_TOKEN'],
            'github_main_branch': GITHUB_MAIN_BRANCH,
            'dvc_remote_name': DVC_REMOTE_DB,
            'dvc_remote_db_url': DVC_REMOTE_DB_URL,
            'minio_url': MINIO_CLUSTER_URL,
            'minio_access_key': MINIO_ACCESS_KEY,
            'minio_secret_key': MINIO_SECRET_KEY,
            'dvc_file_dir': DVC_FILE_DIR,
            'dvc_file_name': DVC_FILE_NAME,
            'model_name': MODEL_NAME,
            'kserve_namespace': KSERVE_NAMESPACE,
            'model_lr': MODEL_LR,
            'model_epochs': MODEL_EPOCHS,
            'model_print_frequency_per_n_epochs': MODEL_PRINT_FREQUENCY_PER_N_EPOCHS,
            'bucket_name': MINIO_MODEL_BUCKET_NAME,
            'minio_model_object_name': MINIO_MODEL_OBJECT_NAME,
            'kserve_svc_acc': KSERVE_SVC_ACC,
            'trigger_type': trigger_type,  # Include the trigger type in the pipeline parameters
            'performance_factor': performance_factor,
            'last_accuracy_object_name': LAST_ACC_OBJECT_NAME,
            'tmp_dir': TEMP_DIR,
            'tmp_file_last_acc': TEMP_FILE_ACC_IN_LAST_RUN,
            'k8s_api_token': K8S_API_TOKEN
        }

        if exec_pipe:
            # Execute the pipeline
            execute_pipeline_run(
                kfp_host=KUBEFLOW_HOST_URL,
                dex_user=DEX_USER,
                dex_pass=DEX_PASS,
                namespace=KSERVE_NAMESPACE,
                job_name=job_name,
                params=pipeline_params,
                pipeline_id=KUBEFLOW_PIPELINE_ID,
                version_id=KUBEFLOW_VERSION_ID,
                svc_acc=KUBEFLOW_SVC_ACC
            )

        # Flash a success message
        flash(flash_msg, 'success')
        
        # Clean up the temporary file after processing
        os.remove(source_csv_path)
        logger.info(f"Temporary CSV file {source_csv_path} has been removed.")
        
        # Redirect to the home page
        return redirect(url_for('home'))
    
@app.route('/inference')
def inference_form():
    """Route to render the inference form page."""
    return render_template('inference_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Route to handle inference requests and return predictions."""
    try:
        # Get the data type and data payload from the request
        data_type = request.json.get('dataType')
        input_data = request.json.get('data')

        # Prepare payload for inference request based on the data type
        payload = {"instances": input_data['instances']} if data_type == "multiple" else {"instances": [input_data]}

        # Send inference request to the model server
        response = requests.post(INFERENCE_URL, headers={"Content-Type": "application/json"}, json=payload)

        # Check for successful response
        if response.status_code == 200:
            predictions = response.json()
            return jsonify({"status": "success", "predictions": predictions}), 200
        else:
            error_message = response.text
            return jsonify({"status": "error", "message": error_message}), response.status_code

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"status": "error", "message": str(e)}), 500


# Define the template folder (you can change the path as needed)
app.template_folder = 'templates'

if __name__ == '__main__':
    app.run(debug=True)
