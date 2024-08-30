from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, get_flashed_messages
from git import Repo
import os
from minio import Minio
import csv
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

# Configuration variables
config = {
    "REPO_URL": os.environ.get('REPO_URL', 'https://github.com/danilonicioka/mlops-workflow.git'),
    "CLONED_DIR": os.environ.get('CLONED_DIR', 'mlops-workflow'),
    "FILE_URL": os.environ.get('FILE_URL', 'https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv'),
    "DVC_FILE_DIR": os.environ.get('DVC_FILE_DIR', 'data/external'),
    "DVC_FILE_NAME": os.environ.get('DVC_FILE_NAME', 'dataset.csv'),
    "BRANCH_NAME": os.environ.get('BRANCH_NAME', 'tests'),
    "BUCKET_NAME": os.environ.get('BUCKET_NAME', 'dvc-data'),
    "MINIO_URL": os.environ.get('MINIO_URL', 'localhost:9000'),
    "ACCESS_KEY": os.environ.get('ACCESS_KEY'),
    "SECRET_KEY": os.environ.get('SECRET_KEY'),
    "REMOTE_NAME": os.environ.get('REMOTE_NAME', 'minio_remote'),
    "GITHUB_USERNAME": os.environ.get('GITHUB_USERNAME'),
    "GITHUB_TOKEN": os.environ.get('GITHUB_TOKEN'),
    "MODEL_NAME": os.environ.get('MODEL_NAME', 'youtubegoes5g'),
    "NAMESPACE": os.environ.get('NAMESPACE', 'kubeflow-user-example-com'),
    "LR": float(os.environ.get('LR', 0.0001)),  # Learning rate, converted to float
    "EPOCHS": int(os.environ.get('EPOCHS', 3500)),  # Number of epochs, converted to int
    "PRINT_FREQUENCY": int(os.environ.get('PRINT_FREQUENCY', 500)),  # Print frequency, converted to int
    "OBJECT_NAME": os.environ.get('OBJECT_NAME', 'model-files'),
    "SVC_ACC": os.environ.get('SVC_ACC', 'sa-minio-kserve'),
    "PIPELINE_ID": os.environ.get('PIPELINE_ID', '7451916e-eee8-4c14-ad5f-8dee5aa61e3b'),
    "VERSION_ID": os.environ.get('VERSION_ID', '264564bb-0ada-4095-920f-ae3bb9d8ca2e'),
    "KFP_HOST": os.environ.get('KFP_HOST', 'http://localhost:8080'),
    "KFP_AUTH_TOKEN": os.environ.get('KFP_AUTH_TOKEN'),  # Token for Kubeflow Pipelines authentication
    "DEX_USER": os.environ.get('DEX_USER'),
    "DEX_PASS": os.environ.get('DEX_PASS'),
    "SVC_ACC_KFP": os.environ.get('SVC_ACC', 'default-editor'),
}

# File paths and commit messages constants
DVC_FILE_PATH_EXT = os.path.join(f"{config['DVC_FILE_DIR']}/{config['DVC_FILE_NAME']}.dvc")
GITIGNORE_PATH = os.path.join(config["DVC_FILE_DIR"], '.gitignore')

COMMIT_MSG_INIT = 'Add .dvc and .gitignore files'
COMMIT_MSG_APPEND = 'Update .dvc file'

TRIGGER_TYPE = '1'

# manage ids
# exp_id = 0

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
    
    # Clone the repository from the specified branch (in this case, 'tests')
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

def commit_and_push_changes(repo, file_paths, commit_message):
    """Commit changes to Git and push them to the 'tests' branch in GitHub."""
    try:
        # Add specified file paths to the Git index
        repo.index.add(file_paths)
        
        # Commit changes with the specified message
        repo.index.commit(commit_message)
        logger.info(f'Successfully committed changes to Git for files: {file_paths}')
        
        # Push changes to the 'tests' branch in GitHub
        origin = repo.remotes.origin
        origin.push(refspec='HEAD:refs/heads/tests')  # Push changes to the 'tests' branch
        
        logger.info('Successfully pushed changes to GitHub on the "tests" branch')
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

    # test the client by listing experiments
    experiments = client.list_experiments(namespace=namespace)
    print(experiments)

    # update exp_id
    # global exp_id
    # exp_id += 1

    """Execute an existing pipeline on Kubeflow."""
    try:
        # Execute the pipeline
        run = client.run_pipeline(
            experiment_id="23d52751-4aeb-4e71-a47e-01c1ced25793",
            job_name=job_name,  # A name for the pipeline run
            params=params,
            pipeline_id=pipeline_id,
            version_id=version_id,
            service_account=svc_acc
        )

        logger.info(f"Pipeline run created successfully: {run.run_id}")
        return run
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        raise

# Routes

@app.route('/')
def home():
    """Route to render the home page with buttons to navigate to /init and /append_csv routes."""
    # Check for flashed messages
    success_message = get_flashed_messages()
    # Render the home.html template with the success message
    return render_template('home.html', success_message=success_message)

@app.route('/init', methods=['GET'])
def init():
    """Initialize the application by cloning the repository, downloading file, and setting up DVC and Git."""
    try:
        # Define the target CSV file path as dataset.csv in the DVC file directory
        target_csv_path = os.path.join(config["CLONED_DIR"], config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])

        # Clone the repository from the tests branch
        repo = clone_repository_with_token(
            config["REPO_URL"], config["CLONED_DIR"], config["BRANCH_NAME"],
            config["GITHUB_USERNAME"], config["GITHUB_TOKEN"]
        )

        # Download the file and save it as the target CSV file
        download_file(config["FILE_URL"], target_csv_path)

        # Initialize DVC and Git repositories
        repo = initialize_dvc_and_repo(config["CLONED_DIR"])

        # Specify the relative path to the target CSV file
        relative_target_csv_path = os.path.join(config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])
        
        # Add the file to DVC using the relative path
        add_file_to_dvc(config["CLONED_DIR"], relative_target_csv_path)
        
        # Commit changes to Git and push them to the 'tests' branch in GitHub using relative paths
        commit_and_push_changes(repo, [DVC_FILE_PATH_EXT, GITIGNORE_PATH], COMMIT_MSG_INIT)

        # Set up Minio client and create a bucket if needed
        client = setup_minio_client(config["MINIO_URL"], config["ACCESS_KEY"], config["SECRET_KEY"], config["BUCKET_NAME"])

        # Configure Minio as the remote DVC repository
        remote_url = f's3://{config["BUCKET_NAME"]}'
        configure_dvc_remote(config["CLONED_DIR"], config["REMOTE_NAME"], remote_url, config["MINIO_URL"], config["ACCESS_KEY"], config["SECRET_KEY"])

        # Push data to remote DVC repository
        push_data_to_dvc(config["CLONED_DIR"], config["REMOTE_NAME"])

        # Flash a success message and redirect to the home page
        flash('Successfully initialized the app, downloaded file as the target CSV file, added data to DVC, committed changes to GitHub, and pushed data to remote DVC repository.')
        return redirect(url_for('home'))

    except Exception as e:
        # Handle and log exceptions
        logger.error(f'Failed to initialize the application: {e}')
        return jsonify({'error': str(e)}), 400

@app.route('/append_csv', methods=['GET', 'POST'])
@handle_dvc_errors
def append_csv():
    """Append data from the source CSV file to the target CSV file named as config["DVC_FILE_NAME"], then update DVC and Git."""
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
            return jsonify({'error': 'Trigger type not provided'}), 400

        logger.info(f"Trigger type received: {trigger_type}")
        
        # Define the path where the uploaded file will be saved
        source_csv_path = os.path.join(config["CLONED_DIR"], 'temp_source.csv')
        
        # Save the uploaded file to a temporary location
        uploaded_file.save(source_csv_path)
        logger.info(f"Uploaded CSV file saved to {source_csv_path}")
        
        # Define the target CSV file path as config["DVC_FILE_NAME"] in the DVC file directory
        target_csv_path = os.path.join(config["CLONED_DIR"], config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])
        
        # Perform a DVC pull to ensure local data is up-to-date with the remote repository
        perform_dvc_pull(config["CLONED_DIR"])
        
        # Append data from the source CSV file to the target CSV file
        append_csv_data(source_csv_path, target_csv_path)
        
        # Open the existing Git repository
        repo = Repo(config["CLONED_DIR"])

        # Specify the relative path to config["DVC_FILE_NAME"]
        relative_target_csv_path = os.path.join(config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])
        
        # Add the appended file to DVC using the relative path
        add_file_to_dvc(config["CLONED_DIR"], relative_target_csv_path)
        
        # Push changes to the remote DVC repository
        push_data_to_dvc(config["CLONED_DIR"], config["REMOTE_NAME"])
        
        # Commit changes to Git and push to GitHub for the updated .dvc file
        commit_and_push_changes(repo, [DVC_FILE_PATH_EXT], COMMIT_MSG_APPEND)

        # Parameters for the pipeline execution
        pipeline_params = {
            'repo_url': config["REPO_URL"],
            'cloned_dir': config["CLONED_DIR"],
            'branch_name': config["BRANCH_NAME"],
            'github_username': config["GITHUB_USERNAME"],
            'github_token': config["GITHUB_TOKEN"],
            'remote_name': config["REMOTE_NAME"],
            'remote_url': f's3://{config["BUCKET_NAME"]}',
            'minio_url': config["MINIO_URL"],
            'access_key': config["ACCESS_KEY"],
            'secret_key': config["SECRET_KEY"],
            'dvc_file_dir': config["DVC_FILE_DIR"],
            'dvc_file_name': config["DVC_FILE_NAME"],
            'model_name': config["MODEL_NAME"],
            'namespace': config["NAMESPACE"],
            'lr': config["LR"],
            'epochs': config["EPOCHS"],
            'print_frequency': config["PRINT_FREQUENCY"],
            'bucket_name': config["BUCKET_NAME"],
            'object_name': config["OBJECT_NAME"],
            'svc_acc': config["SVC_ACC"],
            'trigger_type': trigger_type  # Include the trigger type in the pipeline parameters
        }

        exec_pipe = True
        flash_msg = "Successfully appended data from the uploaded CSV file to the target CSV file, added the file to DVC, pushed changes to the remote repository"
        job_name = "Always retraining trigger job"

        # Determine the job name based on the trigger type
        # if trigger_type == '0':
        #     exec_pipe = False
        # if trigger_type == '1':
        #     flash_msg = f'{flash_msg}, and executed the pipeline.'
        # elif trigger_type == '2':
        #     job_name = "Quantity trigger job"
        # elif trigger_type == '3':
        #     job_name = "Performance trigger job"
        # else:
        #     job_name = "Conditional retraining job"
        #     flash_msg = f'Trigger type {trigger_type} not defined'
        #     exec_pipe = False

        if exec_pipe:
            # Execute the pipeline
            execute_pipeline_run(
                kfp_host=config['KFP_HOST'],
                dex_user=config['DEX_USER'],
                dex_pass=config['DEX_PASS'],
                namespace=config['NAMESPACE'],
                job_name=job_name,
                params=pipeline_params,
                pipeline_id=config['PIPELINE_ID'],
                version_id=config['VERSION_ID'],
                svc_acc=config['SVC_ACC_KFP']
            )

        # Flash a success message
        flash(flash_msg)
        
        # Clean up the temporary file after processing
        os.remove(source_csv_path)
        logger.info(f"Temporary CSV file {source_csv_path} has been removed.")
        
        # Redirect to the home page
        return redirect(url_for('home'))

# Define the template folder (you can change the path as needed)
app.template_folder = 'templates'

if __name__ == '__main__':
    app.run(debug=True)
