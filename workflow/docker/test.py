from flask import Flask, request, jsonify
from git import Repo
import subprocess
import os
from minio import Minio
from minio.error import BucketAlreadyOwnedByYou, BucketAlreadyExists
import csv

app = Flask(__name__)

# Global variables to store the local directory path and the file path added to DVC
cloned_dir = None
dvc_file_path = None

# Route to clone a Git repository, initialize DVC, add a file to DVC, commit changes to Git,
# and configure a Minio bucket as the DVC remote repository.
@app.route('/add_file', methods=['POST'])
def add_file():
    global cloned_dir, dvc_file_path  # Declare the global variables to modify them

    # Get repository URL, branch name, and local directory from request JSON
    repo_url = request.json.get('repo_url')
    branch_name = request.json.get('branch_name', 'main')
    cloned_dir = request.json.get('local_dir', '/tmp/repo')
    file_path = request.json.get('file_path')
    bucket_name = request.json.get('bucket_name', 'dvc-data')
    minio_url = request.json.get('minio_url')  # Example: 'http://minio.local:9000'
    access_key = request.json.get('access_key')
    secret_key = request.json.get('secret_key')
    
    # Clone the Git repository
    try:
        repo = Repo.clone_from(repo_url, cloned_dir, branch=branch_name)
        print(f"Successfully cloned repository from {repo_url} to {cloned_dir}")
    except Exception as e:
        return jsonify({'error': f'Failed to clone repository: {e}'}), 400
    
    # Initialize a DVC repository in the cloned directory
    try:
        dvc_init_result = subprocess.run(['dvc', 'init'], cwd=cloned_dir, capture_output=True, text=True)
        if dvc_init_result.returncode == 0:
            print(f"Successfully initialized DVC in {cloned_dir}")
        else:
            return jsonify({'error': f'DVC init failed: {dvc_init_result.stderr}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to initialize DVC: {e}'}), 400
    
    # Construct the full file path in the cloned repository
    dvc_file_path = os.path.join(cloned_dir, file_path)
    
    # Add the file to DVC
    try:
        result = subprocess.run(['dvc', 'add', dvc_file_path], cwd=cloned_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f'Successfully added {dvc_file_path} to DVC')
        else:
            return jsonify({'error': result.stderr}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to add file to DVC: {e}'}), 400
    
    # Get paths for .dvc and .gitignore files
    dvc_file_path_ext = f'{dvc_file_path}.dvc'
    gitignore_path = os.path.join(cloned_dir, '.gitignore')
    
    # Add the .dvc file and .gitignore file to Git and commit the changes
    try:
        # Add the .dvc file and .gitignore file to Git
        repo.index.add([dvc_file_path_ext, gitignore_path])
        
        # Commit the changes
        repo.index.commit('Add .dvc and .gitignore files')
        print(f'Successfully added and committed {dvc_file_path_ext} and {gitignore_path} to Git')
    except Exception as e:
        return jsonify({'error': f'Failed to commit changes to Git: {e}'}), 400
    
    # Connect to Minio and create a bucket if it doesn't exist
    try:
        # Connect to the Minio client
        client = Minio(
            minio_url,
            access_key=access_key,
            secret_key=secret_key,
            secure=minio_url.startswith('https')  # Check if the URL starts with https
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
            ['dvc', 'remote', 'modify', remote_name, 'endpointurl', minio_url],
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
        
        return jsonify({
            'message': f'Successfully added {dvc_file_path} to DVC, committed changes to Git, configured Minio bucket as DVC remote repository: {remote_name}, and pushed data to DVC remote.'
        }), 200
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
        dvc_pull_result = subprocess.run(['dvc', 'pull'], cwd=cloned_dir, capture_output=True, text=True)
        
        if dvc_pull_result.returncode == 0:
            print("Successfully pulled data from remote DVC repository")
        else:
            return jsonify({'error': f'dvc pull failed: {dvc_pull_result.stderr}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to perform dvc pull: {e}'}), 400
    
    # Append data from the source CSV file to the target CSV file
    try:
        # Read the source CSV file
        with open(source_csv, 'r') as source_file:
            reader = csv.reader(source_file)
            
            # Append the rows from the source file to the target file
            with open(target_csv, 'a', newline='') as target_file:
                writer = csv.writer(target_file)
                
                for row in reader:
                    writer.writerow(row)
        
        print(f"Successfully appended data from {source_csv} to {target_csv}.")
    except Exception as e:
        return jsonify({'error': f'Failed to append data: {e}'}), 400
    
    # Add the appended file to DVC to track the changes
    try:
        dvc_add_result = subprocess.run(['dvc', 'add', target_csv], cwd=cloned_dir, capture_output=True, text=True)
        
        if dvc_add_result.returncode == 0:
            print(f"Successfully added the appended file {target_csv} to DVC.")
        else:
            return jsonify({'error': f'dvc add failed: {dvc_add_result.stderr}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to add file to DVC: {e}'}), 400
    
    # Perform dvc push in the cloned directory to push the changes to the remote repository
    try:
        dvc_push_result = subprocess.run(['dvc', 'push'], cwd=cloned_dir, capture_output=True, text=True)
        
        if dvc_push_result.returncode == 0:
            print("Successfully pushed changes to remote DVC repository.")
        else:
            return jsonify({'error': f'dvc push failed: {dvc_push_result.stderr}'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to push changes to DVC remote: {e}'}), 400
    
    return jsonify({
        'message': f'Successfully appended data from {source_csv} to {target_csv}, added the file to DVC, and pushed changes to remote repository.'
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
