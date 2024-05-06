from flask import Flask, request, jsonify
from git import Repo
import os
from minio import Minio
import csv
import requests
import logging
from dvc.api import DVCFileSystem
import subprocess

# Flask application setup
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration variables
config = {
    "REPO_URL": os.environ.get('REPO_URL', 'https://github.com/danilonicioka/mlops-workflow.git'),
    "CLONED_DIR": os.environ.get('CLONED_DIR', 'mlops-workflow'),
    "FILE_URL": os.environ.get('FILE_URL', 'https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv'),
    "DVC_FILE_DIR": os.environ.get('DVC_FILE_DIR', 'data/external'),
    "DVC_FILE_NAME": os.environ.get('DVC_FILE_NAME', 'init_dataset.csv'),
    "BRANCH_NAME": os.environ.get('BRANCH_NAME', 'main'),
    "BUCKET_NAME": os.environ.get('BUCKET_NAME', 'dvc-data'),
    "MINIO_URL": os.environ.get('MINIO_URL', 'localhost:9000'),
    "ACCESS_KEY": os.environ.get('ACCESS_KEY'),
    "SECRET_KEY": os.environ.get('SECRET_KEY'),
    "REMOTE_NAME": os.environ.get('REMOTE_NAME', 'minio_remote'),
    "GITHUB_USERNAME": os.environ.get('GITHUB_USERNAME'),
    "GITHUB_TOKEN": os.environ.get('GITHUB_TOKEN')
}

# File paths and commit messages constants
DVC_FILE_PATH_EXT = os.path.join(config["CLONED_DIR"], f"{config['DVC_FILE_DIR']}/{config['DVC_FILE_NAME']}.dvc")
GITIGNORE_PATH = os.path.join(config["CLONED_DIR"], config["DVC_FILE_DIR"], '.gitignore')

COMMIT_MSG_INIT = 'Add .dvc and .gitignore files'
COMMIT_MSG_APPEND = 'Update .dvc file'

# Helper functions

def handle_dvc_errors(func):
    """Decorator to handle DVC errors and return appropriate error responses."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f'DVC operation failed: {e}')
            return jsonify({'error': str(e)}), 400
    return wrapper

def setup_minio_client(minio_url, access_key, secret_key, bucket_name):
    """Create a Minio client and ensure the bucket exists."""
    client = Minio(
        f'http://{minio_url}',
        access_key=access_key,
        secret_key=secret_key,
        secure=False  # Minio is using HTTP on localhost:9000
    )
    
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        logger.info(f"Successfully created bucket: {bucket_name}")
    else:
        logger.info(f"Bucket {bucket_name} already exists")

    return client

def initialize_dvc_and_repo(cloned_dir):
    """Initialize the Git repository and DVC filesystem."""
    # Initialize the DVC repository using subprocess
    subprocess.run(['dvc', 'init'], cwd=cloned_dir, capture_output=True, text=True, check=True)
    
    # Load Git repository and DVCFileSystem
    repo = Repo(cloned_dir)
    dvc_fs = DVCFileSystem(url=cloned_dir)
    
    logger.info(f"Successfully initialized DVC in {cloned_dir}")
    return repo, dvc_fs

def clone_repository(repo_url, cloned_dir, branch_name):
    """Clone the Git repository."""
    try:
        repo = Repo.clone_from(repo_url, cloned_dir, branch=branch_name)
        logger.info(f"Successfully cloned repository from {repo_url} to {cloned_dir}")
        return repo
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        raise

def download_file(file_url, local_file_path):
    """Download the file from a given URL."""
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        with open(local_file_path, 'wb') as local_file:
            local_file.write(response.content)
        logger.info(f"Successfully downloaded file from {file_url} to {local_file_path}")
    except requests.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        raise

def add_file_to_dvc_fs(dvc_fs, dvc_file_path):
    """Add a file to DVC."""
    try:
        dvc_fs.put(dvc_file_path, dvc_file_path)
        logger.info(f'Successfully added {dvc_file_path} to DVC')
    except Exception as e:
        logger.error(f'Failed to add file to DVC: {e}')
        raise

def commit_and_push_changes(repo, file_paths, commit_message):
    """Commit changes to Git and push them to GitHub."""
    try:
        # Add the specified file paths to the Git index
        repo.index.add(file_paths)
        
        # Commit the changes with the specified commit message
        repo.index.commit(commit_message)
        logger.info(f'Successfully committed changes to Git for files: {file_paths}')
        
        # Push the changes to GitHub
        origin = repo.remotes.origin
        github_username = config["GITHUB_USERNAME"]
        github_token = config["GITHUB_TOKEN"]
        origin.set_url(f"https://{github_username}:{github_token}@{config['REPO_URL']}")
        origin.push()
        logger.info('Successfully pushed changes to GitHub repository')
    except Exception as e:
        logger.error(f'Failed to commit changes to Git or push to GitHub: {e}')
        raise

def configure_dvc_remote(dvc_fs, remote_name, remote_url, minio_url, access_key, secret_key):
    """Configure the Minio bucket as the DVC remote repository."""
    try:
        dvc_fs.remote_add(remote_name, remote_url)
        dvc_fs.remote_modify(remote_name, 'endpointurl', f'http://{minio_url}')
        dvc_fs.remote_modify(remote_name, 'access_key_id', access_key)
        dvc_fs.remote_modify(remote_name, 'secret_access_key', secret_key)
        logger.info(f"Successfully configured Minio bucket as DVC remote repository: {remote_name}")
    except Exception as e:
        logger.error(f'Failed to configure DVC remote: {e}')
        raise

def push_data_to_dvc_fs(dvc_fs, remote_name):
    """Push data to the remote DVC repository."""
    try:
        dvc_fs.push(remote_name)
        logger.info("Successfully pushed data to remote DVC repository")
    except Exception as e:
        logger.error(f'dvc push failed: {e}')
        raise

def perform_dvc_pull(cloned_dir):
    """Perform DVC pull to synchronize local data with the remote repository."""
    result = subprocess.run(['dvc', 'pull'], cwd=cloned_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f'dvc pull failed: {result.stderr}')
    logger.info("Successfully pulled data from remote DVC repository")

def append_csv_data(source_csv, target_csv):
    """Append data from the source CSV file to the target CSV file."""
    with open(source_csv, 'r') as source_file:
        reader = csv.reader(source_file)
        
        # Append rows from the source file to the target file
        with open(target_csv, 'a', newline='') as target_file:
            writer = csv.writer(target_file)
            for row in reader:
                writer.writerow(row)
    logger.info(f"Successfully appended data from {source_csv} to {target_csv}")

# Routes

@app.route('/')
def home():
    """Route to return a welcome message."""
    return "Hello from Flask app"

@app.route('/init', methods=['GET'])
def init():
    # Define the local file path
    local_file_path = os.path.join(config["CLONED_DIR"], config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])

    # Clone the repository
    repo = clone_repository(config["REPO_URL"], config["CLONED_DIR"], config["BRANCH_NAME"])

    # Download file
    download_file(config["FILE_URL"], local_file_path)

    # Initialize DVC and Git repositories
    repo, dvc_fs = initialize_dvc_and_repo(config["CLONED_DIR"])

    # Add file to DVC filesystem
    add_file_to_dvc_fs(dvc_fs, local_file_path)

    # Commit changes to Git and push to GitHub
    commit_and_push_changes(repo, [DVC_FILE_PATH_EXT, GITIGNORE_PATH], COMMIT_MSG_INIT)

    # Set up Minio client and create bucket if needed
    client = setup_minio_client(config["MINIO_URL"], config["ACCESS_KEY"], config["SECRET_KEY"], config["BUCKET_NAME"])

    # Configure Minio as the remote DVC repository
    remote_url = f's3://{config["BUCKET_NAME"]}'
    configure_dvc_remote(dvc_fs, config["REMOTE_NAME"], remote_url, config["MINIO_URL"], config["ACCESS_KEY"], config["SECRET_KEY"])

    # Push data to remote DVC repository
    push_data_to_dvc_fs(dvc_fs, config["REMOTE_NAME"])

    return jsonify({
        'message': 'Successfully initialized the app, downloaded file, added data to DVC, committed changes to GitHub, and pushed data to remote DVC repository.'
    }), 200

@app.route('/append_csv', methods=['POST'])
@handle_dvc_errors
def append_csv():
    # Get the source CSV file path from the request JSON
    source_csv = request.json.get('source_csv')
    
    # Specify the target CSV file path as the file added to DVC in the `/init` route
    target_csv = os.path.join(config["CLONED_DIR"], config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])
    
    # Perform DVC pull to ensure local data is up-to-date with the remote repository
    perform_dvc_pull(config["CLONED_DIR"])
    
    # Append data from the source CSV file to the target CSV file
    append_csv_data(source_csv, target_csv)
    
    # Initialize DVC and Git repositories
    repo, dvc_fs = initialize_dvc_and_repo(config["CLONED_DIR"])

    # Add the appended file to DVC
    add_file_to_dvc_fs(dvc_fs, target_csv)
    
    # Push changes to the remote DVC repository
    push_data_to_dvc_fs(dvc_fs, config["REMOTE_NAME"])

    # Commit changes to Git and push to GitHub for the updated .dvc file
    commit_and_push_changes(repo, [DVC_FILE_PATH_EXT], COMMIT_MSG_APPEND)

    return jsonify({
        'message': f'Successfully appended data from {source_csv} to {target_csv}, added the file to DVC, and pushed changes to remote repository and GitHub.'
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
