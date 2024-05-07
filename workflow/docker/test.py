from flask import Flask, request, jsonify
from git import Repo
import os
from minio import Minio
import csv
import requests
import logging
from subprocess import run, CalledProcessError

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
    "BRANCH_NAME": os.environ.get('BRANCH_NAME', 'tests'),
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

# Constants for commit messages
COMMIT_MSG_INIT = 'Add .dvc and .gitignore files'
COMMIT_MSG_APPEND = 'Update .dvc file'

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

def setup_minio_client(minio_url, access_key, secret_key, bucket_name):
    """Create a Minio client and ensure the bucket exists."""
    # Initialize Minio client
    client = Minio(
        f'http://{minio_url}',
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

def clone_repository(repo_url, cloned_dir, branch_name):
    """Clone the Git repository from the specified URL and branch."""
    try:
        # Clone the repository
        repo = Repo.clone_from(repo_url, cloned_dir, branch=branch_name)
        logger.info(f"Successfully cloned repository from {repo_url} to {cloned_dir}")
        return repo
    except Exception as e:
        # Log and raise any cloning errors
        logger.error(f"Failed to clone repository: {e}")
        raise

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
    """Commit changes to Git and push them to GitHub."""
    try:
        # Add specified file paths to the Git index
        repo.index.add(file_paths)
        
        # Commit changes with the specified message
        repo.index.commit(commit_message)
        logger.info(f'Successfully committed changes to Git for files: {file_paths}')
        
        # Push changes to GitHub
        origin = repo.remotes.origin
        github_username = config["GITHUB_USERNAME"]
        github_token = config["GITHUB_TOKEN"]
        
        # Set the remote URL only if the origin URL doesn't already contain the username and token
        origin_url = origin.url
        # Check if the repository URL starts with 'https://'
        if not origin_url.startswith('https://'):
            origin.set_url(f"https://{github_username}:{github_token}@{config['REPO_URL']}")
        
        # Push changes to GitHub
        origin.push()
        logger.info('Successfully pushed changes to GitHub repository')
    except Exception as e:
        # Log and raise any errors
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
        # Run the `dvc push` command to push data
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
    # Run the `dvc pull` command
    run(['dvc', 'pull'], cwd=cloned_dir, capture_output=True, text=True, check=True)
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
    """Initialize the application by cloning the repository, downloading file, and setting up DVC and Git."""
    # Define the local file path
    local_file_path = os.path.join(config["CLONED_DIR"], config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])

    # Clone the repository
    repo = clone_repository(config["REPO_URL"], config["CLONED_DIR"], config["BRANCH_NAME"])

    # Download the file
    download_file(config["FILE_URL"], local_file_path)

    # Initialize DVC and Git repositories
    repo = initialize_dvc_and_repo(config["CLONED_DIR"])

    # Calculate the relative file path to add to DVC
    relative_file_path = os.path.join(config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])

    # Add the file to DVC using the relative file path
    add_file_to_dvc(config["CLONED_DIR"], relative_file_path)

    # Use relative paths for the `.dvc` and `.gitignore` files when committing and pushing changes to Git
    relative_dvc_file_path = f"{config['DVC_FILE_DIR']}/{config['DVC_FILE_NAME']}.dvc"
    relative_gitignore_path = f"{config['DVC_FILE_DIR']}/.gitignore"

    # Commit changes to Git and push to GitHub using relative paths
    commit_and_push_changes(repo, [relative_dvc_file_path, relative_gitignore_path], COMMIT_MSG_INIT)

    # Set up Minio client and create a bucket if needed
    client = setup_minio_client(config["MINIO_URL"], config["ACCESS_KEY"], config["SECRET_KEY"], config["BUCKET_NAME"])

    # Configure Minio as the remote DVC repository
    remote_url = f's3://{config["BUCKET_NAME"]}'
    configure_dvc_remote(config["CLONED_DIR"], config["REMOTE_NAME"], remote_url, config["MINIO_URL"], config["ACCESS_KEY"], config["SECRET_KEY"])

    # Push data to remote DVC repository
    push_data_to_dvc(config["CLONED_DIR"], config["REMOTE_NAME"])

    return jsonify({
        'message': 'Successfully initialized the app, downloaded file, added data to DVC, committed changes to GitHub, and pushed data to remote DVC repository.'
    }), 200

@app.route('/append_csv', methods=['POST'])
@handle_dvc_errors
def append_csv():
    """Append data from the source CSV file to the target CSV file, then update DVC and Git."""
    # Get the source CSV file path from the request JSON
    source_csv = request.json.get('source_csv')
    
    # Specify the target CSV file path as the file added to DVC in the `/init` route
    target_csv = os.path.join(config["CLONED_DIR"], config["DVC_FILE_DIR"], config["DVC_FILE_NAME"])
    
    # Perform a DVC pull to ensure local data is up-to-date with the remote repository
    perform_dvc_pull(config["CLONED_DIR"])
    
    # Append data from the source CSV file to the target CSV file
    append_csv_data(source_csv, target_csv)
    
    # Initialize DVC and Git repositories to get the `repo` object
    repo = initialize_dvc_and_repo(config["CLONED_DIR"])
    
    # Add the appended file to DVC
    add_file_to_dvc(config["CLONED_DIR"], target_csv)
    
    # Push changes to the remote DVC repository
    push_data_to_dvc(config["CLONED_DIR"], config["REMOTE_NAME"])

    # Commit changes to Git and push to GitHub for the updated .dvc file
    commit_and_push_changes(repo, [DVC_FILE_PATH_EXT], COMMIT_MSG_APPEND)

    return jsonify({
        'message': f'Successfully appended data from {source_csv} to {target_csv}, added the file to DVC, and pushed changes to remote repository and GitHub.'
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
