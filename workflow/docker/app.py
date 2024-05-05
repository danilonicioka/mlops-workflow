from flask import Flask, request, jsonify
from git import Repo
import subprocess
import os
from minio import Minio
from minio.error import BucketAlreadyOwnedByYou, BucketAlreadyExists
import csv
import requests

app = Flask(__name__)

# Global variables to store the local directory path, DVC file directory, and DVC file name
cloned_dir = 'mlops-workflow'  # Define cloned_dir as 'mlops-workflow'
dvc_file_dir = 'data/external'  # Define dvc_file_dir as 'data/external'
dvc_file_name = 'init_dataset.csv'  # Define dvc_file_name as 'init_dataset.csv'

# Route to return "Hello from Flask app"
@app.route('/')
def home():
    return "Hello from Flask app"

# Change the /add_file route to /init
@app.route('/init', methods=['POST'])
def init():
    global cloned_dir, dvc_file_dir, dvc_file_name  # Declare the global variables to modify them

    # Define the repository URL as specified
    repo_url = 'https://github.com/danilonicioka/mlops-workflow.git'
    
    # Define the file URL as specified
    file_url = 'https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv'
    
    # Define the branch name and bucket name
    branch_name = 'main'  # Define the branch name as 'main'
    bucket_name = 'dvc-data'  # Define the bucket name as 'dvc-data'
    
    # Define the Minio URL as specified
    minio_url = 'localhost:9000'  # Define the Minio URL as 'localhost:9000'
    
    # Get Minio credentials from environment variables
    access_key = os.environ.get('MINIO_ACCESS_KEY')
    secret_key = os.environ.get('MINIO_SECRET_KEY')
    
    # Clone the Git repository
    try:
        repo = Repo.clone_from(repo_url, cloned_dir, branch=branch_name)
        print(f"Successfully cloned repository from {repo_url} to {cloned_dir}")
    except Exception as e:
        return jsonify({'error': f'Failed to clone repository: {e}'}), 400
    
    # Download the file
    try:
        # Define the full path of the file in the cloned directory
        local_file_path = os.path.join(cloned_dir, dvc_file_dir, dvc_file_name)
        
        # Download the file from the provided URL
        response = requests.get(file_url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the file to the local file path
            with open(local_file_path, 'wb') as local_file:
                local_file.write(response.content)
            print(f"Successfully downloaded file from {file_url} to {local_file_path}")
        else:
            return jsonify({'error': f'Failed to download file: {response.status_code}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to download file: {e}'}), 400
    
    # Initialize a DVC repository in the cloned directory
    try:
        dvc_init_result = subprocess.run(['dvc', 'init'], cwd=cloned_dir, capture_output=True, text=True)
        if dvc_init_result.returncode == 0:
            print(f"Successfully initialized DVC in {cloned_dir}")
        else:
            return jsonify({'error': f'DVC init failed: {dvc_init_result.stderr}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to initialize DVC: {e}'}), 400
    
    # Add the file to DVC
    try:
        # Define the full DVC file path by combining the directory and file name
        dvc_file_path = os.path.join(dvc_file_dir, dvc_file_name)
        
        # Add the file to DVC
        result = subprocess.run(['dvc', 'add', dvc_file_path], cwd=cloned_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f'Successfully added {dvc_file_path} to DVC')
        else:
            return jsonify({'error': result.stderr}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to add file to DVC: {e}'}), 400
    
    # Add the .dvc and .gitignore files to Git and commit the changes
    try:
        # Get the full path for the .dvc file (including cloned directory)
        dvc_file_path_ext = os.path.join(cloned_dir, f"{dvc_file_path}.dvc")
        
        # Get the full path for the .gitignore file (including cloned directory and DVC file directory)
        gitignore_path = os.path.join(cloned_dir, dvc_file_dir, '.gitignore')
        
        # Add the .dvc file and .gitignore file to Git
        repo.index.add([dvc_file_path_ext, gitignore_path])
        
        # Commit the changes
        repo.index.commit('Add .dvc and .gitignore files')
        print(f'Successfully added and committed {dvc_file_path_ext} and {gitignore_path} to Git')
        
        # Get GitHub credentials from environment variables
        github_username = os.environ.get('GITHUB_USERNAME')
        github_token = os.environ.get('GITHUB_TOKEN')
        
        # Configure the repository to use the provided credentials for pushing to GitHub
        origin = repo.remotes.origin
        origin.set_url(f"https://{github_username}:{github_token}@github.com/danilonicioka/mlops-workflow.git")
        
        # Push the changes to GitHub
        origin.push()
        print(f'Successfully pushed changes to GitHub repository')
    
    except Exception as e:
        return jsonify({'error': f'Failed to commit changes to Git or push to GitHub: {e}'}), 400
    
    # Connect to Minio and create a bucket if it doesn't exist
    try:
        # Connect to the Minio client
        client = Minio(
            f'http://{minio_url}',
            access_key=access_key,
            secret_key=secret_key,
            secure=False  # Specify False since Minio is not using HTTPS on localhost:9000
        )
        
        # Create the bucket if it doesn't exist
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Successfully created bucket: {bucket_name}")
        else:
            print(f"Bucket {bucket_name} already exists")
    except (BucketAlreadyOwnedByYou, BucketAlreadyExists):
        print(f"Bucket {bucket_name} already exists")
    except Exception as e:
        return jsonify({'error': f'Failed to connect to Minio or create bucket: {e}'}), 400
    
    # Configure the Minio bucket as the remote repository for DVC data
    try:
        remote_name = 'minio_remote'  # Name of the DVC remote
        remote_url = f's3://{bucket_name}'  # Use s3 URL format
        
        # Configure the remote repository
        subprocess.run(
            ['dvc', 'remote', 'add', remote_name, remote_url],
            cwd=cloned_dir,
            capture_output=True,
            text=True
        )
        
        # Set the remote repository configuration
        subprocess.run(
            ['dvc', 'remote', 'modify', remote_name, 'endpointurl', f'http://{minio_url}'],
            cwd=cloned_dir,
            capture_output=True,
            text=True
        )
        
        subprocess.run(
            ['dvc', 'remote', 'modify', remote_name, 'access_key_id', access_key],
            cwd=cloned_dir,
            capture_output=True,
            text=True
        )
        
        subprocess.run(
            ['dvc', 'remote', 'modify', remote_name, 'secret_access_key', secret_key],
            cwd=cloned_dir,
            capture_output=True,
            text=True
        )
        
        print(f"Successfully configured Minio bucket as DVC remote repository: {remote_name}")
        
        # Perform dvc push to push data to the remote repository
        dvc_push_result = subprocess.run(['dvc', 'push'], cwd=cloned_dir, capture_output=True, text=True)
        
        if dvc_push_result.returncode == 0:
            print("Successfully pushed data to remote DVC repository")
        else:
            return jsonify({'error': f'dvc push failed: {dvc_push_result.stderr}'}), 400
        
        # After pushing data to DVC repository, add, commit, and push the changes of the updated .dvc file to the GitHub repository
        # Add the updated .dvc file to Git
        repo.index.add([dvc_file_path_ext])
        
        # Commit the changes
        repo.index.commit('Update .dvc file')
        print(f'Successfully committed the updated .dvc file to Git')
        
        # Push the changes to GitHub
        origin.push()
        print('Successfully pushed changes to GitHub repository')
    
    except Exception as e:
        return jsonify({'error': f'Failed to configure DVC remote: {e}'}), 400
    
    return jsonify({
        'message': f'Successfully initialized the app, downloaded file, added data to DVC, committed changes to GitHub, and pushed data to remote DVC repository.'
    }), 200

if __name__ == '__main__':
    app.run(debug=True)