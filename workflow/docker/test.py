from flask import Flask, request, jsonify
from git import Repo
import os
from minio import Minio
from minio.error import BucketAlreadyOwnedByYou, BucketAlreadyExists
import requests
import logging
from dvc.api import DVC
import requests

app = Flask(__name__)
app.config.from_pyfile('config.py')  # Load configuration from a config file

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to initialize and clone repository
def clone_repository(repo_url, cloned_dir, branch_name):
    try:
        repo = Repo.clone_from(repo_url, cloned_dir, branch=branch_name)
        logger.info(f"Successfully cloned repository from {repo_url} to {cloned_dir}")
        return repo
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        raise

# Function to download file
def download_file(file_url, local_file_path):
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        with open(local_file_path, 'wb') as local_file:
            local_file.write(response.content)
        logger.info(f"Successfully downloaded file from {file_url} to {local_file_path}")
    except requests.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        raise

# Function to initialize DVC repository
def initialize_dvc(cloned_dir):
    try:
        dvc = DVC(cloned_dir)
        dvc.init()
        logger.info(f"Successfully initialized DVC in {cloned_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize DVC: {e}")
        raise

# Function to add file to DVC
def add_file_to_dvc(dvc, dvc_file_path):
    try:
        dvc.add(dvc_file_path)
        logger.info(f'Successfully added {dvc_file_path} to DVC')
    except Exception as e:
        logger.error(f'Failed to add file to DVC: {e}')
        raise

# Function to commit changes to Git and push to GitHub
def commit_and_push_changes(repo, dvc_file_path_ext, gitignore_path):
    try:
        repo.index.add([dvc_file_path_ext, gitignore_path])
        repo.index.commit('Add .dvc and .gitignore files')
        logger.info(f'Successfully added and committed {dvc_file_path_ext} and {gitignore_path} to Git')

        origin = repo.remotes.origin
        origin.set_url(f"https://{os.environ.get('GITHUB_USERNAME')}:{os.environ.get('GITHUB_TOKEN')}@{app.config['REPO_URL']}")
        origin.push()
        logger.info(f'Successfully pushed changes to GitHub repository')
    except Exception as e:
        logger.error(f'Failed to commit changes to Git or push to GitHub: {e}')
        raise

# Function to connect to Minio and create a bucket
def create_minio_bucket(client, bucket_name):
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"Successfully created bucket: {bucket_name}")
        else:
            logger.info(f"Bucket {bucket_name} already exists")
    except (BucketAlreadyOwnedByYou, BucketAlreadyExists):
        logger.info(f"Bucket {bucket_name} already exists")
    except Exception as e:
        logger.error(f'Failed to connect to Minio or create bucket: {e}')
        raise

# Function to configure DVC remote repository
def configure_dvc_remote(dvc, remote_name, remote_url, minio_url, access_key, secret_key):
    try:
        dvc.remote_add(remote_name, remote_url)
        dvc.remote_modify(remote_name, 'endpointurl', f'http://{minio_url}')
        dvc.remote_modify(remote_name, 'access_key_id', access_key)
        dvc.remote_modify(remote_name, 'secret_access_key', secret_key)
        logger.info(f"Successfully configured Minio bucket as DVC remote repository: {remote_name}")
        return remote_name
    except Exception as e:
        return jsonify({'error': f'Failed to configure DVC remote: {e}'}), 400

# Route to append data from one CSV file to the target CSV file (the file added to DVC)
@app.route('/append_csv', methods=['POST'])
def append_csv():
    global cloned_dir, dvc_file_path  # Access the global variables

    # Get the source CSV file path from the request JSON
    source_csv = request.json.get('source_csv')
    
    # Use the target CSV file path as the file added to DVC in the `/add_file` route
    target_csv = dvc_file_path
    
    # Perform dvc pull in the local directory to ensure local data is up-to-date with the remote repository
    try:
        dvc.push(remote_name)
        logger.info("Successfully pushed data to remote DVC repository")
    except Exception as e:
        logger.error(f'dvc push failed: {e}')
        raise

# Function to update .dvc file and push changes to GitHub
def update_dvc_file_and_push(repo, dvc_file_path_ext, origin):
    try:
        repo.index.add([dvc_file_path_ext])
        repo.index.commit('Update .dvc file')
        logger.info(f'Successfully committed the updated .dvc file to Git')
        origin.push()
        logger.info('Successfully pushed changes to GitHub repository')
    except Exception as e:
        logger.error(f'Failed to update .dvc file or push changes to GitHub: {e}')
        raise

@app.route('/init', methods=['POST'])
def init():
    # Load global variables from configuration
    repo_url = app.config['REPO_URL']
    cloned_dir = app.config['CLONED_DIR']
    file_url = app.config['FILE_URL']
    dvc_file_dir = app.config['DVC_FILE_DIR']
    dvc_file_name = app.config['DVC_FILE_NAME']
    branch_name = app.config['BRANCH_NAME']
    bucket_name = app.config['BUCKET_NAME']
    minio_url = app.config['MINIO_URL']
    access_key = app.config['ACCESS_KEY']
    secret_key = app.config['SECRET_KEY']
    remote_name = app.config['REMOTE_NAME']

    # Define the file path
    local_file_path = os.path.join(cloned_dir, dvc_file_dir, dvc_file_name)
    dvc_file_path = os.path.join(dvc_file_dir, dvc_file_name)
    dvc_file_path_ext = os.path.join(cloned_dir, f"{dvc_file_path}.dvc")
    gitignore_path = os.path.join(cloned_dir, dvc_file_dir, '.gitignore')

    # Clone repository
    repo = clone_repository(repo_url, cloned_dir, branch_name)

    # Download file
    download_file(file_url, local_file_path)

    # Initialize DVC
    dvc = DVC(cloned_dir)
    initialize_dvc(cloned_dir)

    # Add file to DVC
    add_file_to_dvc(dvc, dvc_file_path)

    # Commit changes to Git and push to GitHub
    commit_and_push_changes(repo, dvc_file_path_ext, gitignore_path)

    # Connect to Minio and create bucket
    client = Minio(
        f'http://{minio_url}',
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    create_minio_bucket(client, bucket_name)

    # Configure DVC remote repository
    remote_url = f's3://{bucket_name}'
    configure_dvc_remote(dvc, remote_name, remote_url, minio_url, access_key, secret_key)

    # Push data to remote DVC repository
    push_data_to_dvc(dvc, remote_name)

    # Update .dvc file and push changes to GitHub
    update_dvc_file_and_push(repo, dvc_file_path_ext, repo.remotes.origin)

    return jsonify({
        'message': f'Successfully appended data from {source_csv} to {target_csv}, added the file to DVC, and pushed changes to remote repository.'
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
