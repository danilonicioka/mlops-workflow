import os
from dotenv import load_dotenv
import os
import requests
import logging
import pandas as pd
import numpy as np
import torch
import numpy as np
from torch import nn
from torch import nn

# Load environment variables from env file
load_dotenv('env')

# Github variables
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO_URL = "https://github.com/danilonicioka/mlops-workflow.git"
GITHUB_CLONED_DIR = "mlops-workflow"
GITHUB_DVC_BRANCH = "dvc"
GITHUB_MAIN_BRANCH = "main"

# Kubeflow variables
# KUBEFLOW_PIPELINE_NAME = "mlops"
# KUBEFLOW_HOST_URL = "http://ml-pipeline.kubeflow:8888"  # KFP host URL
# KUBEFLOW_PIPELINE_ID="7451916e-eee8-4c14-ad5f-8dee5aa61e3b"
# with open(os.environ['KF_PIPELINES_SA_TOKEN_PATH'], "r") as f:
#     KUBEFLOW_TOKEN = f.read()

# DVC variables
DVC_REMOTE_DB = "minio_remote"
DVC_REMOTE_DB_URL = "s3://dvc-data"
DVC_FILE_DIR = 'data/external'
DVC_FILE_NAME = 'dataset.csv'

# MinIO variables
MINIO_URL = "minio-service.kubeflow:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_MODEL_BUCKET_NAME = "model-files"
MINIO_MODEL_OBJECT_NAME = "model-store/youtubegoes5g/model.pt"

# Triggers variables
TRIGGER_TYPE = '1'
PERFORMANCE_FACTOR = 0.05
# Temp dir and files to save accuracy for trigger 3
TEMP_DIR = "tmp"
TEMP_FILE_ACC_IN_LAST_RUN = "accuracy_in_last_run.txt"
LAST_ACC_OBJECT_NAME = "accuracy-score/last_acc.txt"

# Model variables
MODEL_LR = 0.0001
MODEL_EPOCHS = 3500
MODEL_PRINT_FREQUENCY_PER_N_EPOCHS = 500
MODEL_NAME = "youtubegoes5g"

# Kserve variables
#MODEL_FRAMEWORK = "pytorch"
KSERVE_NAMESPACE = "kubeflow-user-example-com"
KSERVE_SVC_ACC = "sa-minio-kserve"
#MODEL_URI = "pvc://model-store-claim"
#MODEL_URI = "minio-service.kubeflow:9000/model-files"

# Model archiver gen vars
MODEL_STORE_POD_NAME = "model-store-pod"
MODEL_STORE_POD_CONTAINER_NAME = "model-store"
MAR_POD_NAME = "margen-pod"
MAR_POD_CONTAINER_NAME = "margen-container"
MAR_OBJECT_NAME = "model-store/youtubegoes5g.mar"
K8S_API_TOKEN = os.getenv("K8S_API_TOKEN")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_INIT_DATASET_URL = 'https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv'

file_url = MODEL_INIT_DATASET_URL
local_file_path = DVC_FILE_NAME

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

path_to_pt = "../model-archiver/model-store/youtubegoes5g/model.pt"

model = InterruptionModel()
model.load_state_dict(torch.load(path_to_pt, weights_only=True))
model.eval()

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

# get sample
# Step 1: Read the first row of the CSV file
sample = pd.read_csv('dataset.csv', nrows=1)

# Capture the value of the 'Stall' column before dropping it
stall_value = sample['Stall'].values[0] if 'Stall' in sample.columns else None
print("Stall Value:", stall_value)

# Step 2: Drop the 'Stall', 'ID', 'Quality', and 'Time' columns
sample = sample.drop(columns=['Stall', 'ID', 'Quality', 'Time'], errors='ignore')

# Step 3: Replace ' ', '-', and np.nan with 0
sample = sample.replace([' ', '-', np.nan], 0)

# Convert all columns to float
sample = sample.astype(float)

# Step 4: Extract the first sample as a NumPy array without the column names
first_sample = sample.values.flatten()  # Use flatten() to get a 1D array

# Step 5: Convert the first sample to a PyTorch tensor
first_sample_tensor = torch.tensor(first_sample, dtype=torch.float32)

# Display the tensor
print("First Sample Tensor:", first_sample_tensor)

with torch.no_grad():
    y_pred = torch.round(torch.sigmoid(model(first_sample_tensor))).squeeze()

if device == "cuda":
    prediction = y_pred.cpu().numpy() #if it is cuda, then this, otherwise y_pred.numpy()
else:
    prediction = y_pred.numpy()

if prediction == 0:
    result = "No Stall"
elif prediction == 1:
    result = "Stall"
    
print(result)