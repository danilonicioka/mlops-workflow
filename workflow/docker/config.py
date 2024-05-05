import os

# Application Configuration

# Git Repository URL
REPO_URL = 'https://github.com/danilonicioka/mlops-workflow.git'

# Cloned Repository Directory
CLONED_DIR = 'mlops-workflow'

# File to be downloaded (URL)
FILE_URL = 'https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv'

# DVC Configuration
DVC_FILE_DIR = 'data/external'  # Directory where the file will be saved
DVC_FILE_NAME = 'init_dataset.csv'  # Name of the file to be saved

# Branch name for Git operations
BRANCH_NAME = 'main'

# Minio Configuration
MINIO_URL = 'localhost:9000'  # URL for Minio instance

# Access keys for Minio
ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')  # Retrieve from environment variables
SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')  # Retrieve from environment variables

# Minio Bucket name
BUCKET_NAME = 'dvc-data'

# DVC Remote Name
REMOTE_NAME = 'minio_remote'