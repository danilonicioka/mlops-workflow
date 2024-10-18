import os
from dotenv import load_dotenv

# Load environment variables from env file
load_dotenv('env')

# Github variables
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO_URL = "https://github.com/danilonicioka/mlops-workflow.git"
GITHUB_CLONED_DIR = "mlops-workflow"
GITHUB_DVC_BRANCH = "dvc"
GITHUB_MAIN_BRANCH = "main"

# # Kubeflow variables
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

import os
import requests
import logging
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_INIT_DATASET_URL = 'https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv'

file_url = MODEL_INIT_DATASET_URL
local_file_path = DVC_FILE_NAME

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

import pandas as pd
import numpy as np
from mlem.api import save
import torch
import numpy as np
from torch import nn

# load data
df = pd.read_csv(local_file_path)

df = df.replace([' ', '-',np.nan], np.nan)

# Selective columns for mean calculation
columns_to_convert = ['CQI1', 'CQI2', 'CQI3', 'cSTD CQI',
       'cMajority', 'c25 P', 'c50 P', 'c75 P', 'RSRP1', 'RSRP2', 'RSRP3',
       'pMajority', 'p25 P', 'p50 P', 'p75 P', 'RSRQ1', 'RSRQ2', 'RSRQ3',
       'qMajority', 'q25 P', 'q50 P', 'q75 P', 'SNR1', 'SNR2', 'SNR3',
       'sMajority', 's25 P', 's50 P', 's75 P']
df[columns_to_convert] = df[columns_to_convert].astype(float)

# Replace np.nan with mean values for selective columns
df[columns_to_convert] = df[columns_to_convert].fillna(df[columns_to_convert].mean())

df['Stall'].replace('Yes', 1, inplace=True)
df['Stall'].replace('No', 0, inplace=True)

X = df[['CQI1', 'CQI2', 'CQI3', 'cSTD CQI',
       'cMajority', 'c25 P', 'c50 P', 'c75 P', 'RSRP1', 'RSRP2', 'RSRP3',
       'pMajority', 'p25 P', 'p50 P', 'p75 P', 'RSRQ1', 'RSRQ2', 'RSRQ3',
       'qMajority', 'q25 P', 'q50 P', 'q75 P', 'SNR1', 'SNR2', 'SNR3',
       'sMajority', 's25 P', 's50 P', 's75 P']].values

y = df['Stall'].values

import sklearn
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build model with non-linear activation function
from torch import nn
class InteruptionModel(nn.Module):
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

model = InteruptionModel().to(device)

# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

# Fit the model
torch.manual_seed(42)
epochs = 3500

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model(X_train).squeeze()

    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels

    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model.eval()
    with torch.no_grad():
      # 1. Forward pass
        test_logits = model(X_test).squeeze()
        #print(test_logits.shape)
        test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
        # 2. Calcuate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)


    # Print out what's happening
    if epoch % 500 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

model.eval()
with torch.no_grad():
     y_preds = torch.round(torch.sigmoid(model(X_test))).squeeze()

if device == "cuda":
  predictions = y_preds.cpu().numpy() #if it is cuda, then this, otherwise y_pred.numpy()
  true_labels = y_test.cpu().numpy()
else:
  predictions = y_preds.numpy()
  true_labels = y_test.numpy()

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,fbeta_score

print("=== Confusion Matrix ===")
print(confusion_matrix(true_labels, predictions))
print('\n')


print("=== Score ===")
accuracy = accuracy_score(true_labels, predictions)
print('Accuracy: %f' % accuracy)

precision = precision_score(true_labels,  predictions, average='weighted')
print('Precision: %f' % precision)
recall = recall_score(true_labels, predictions, average='weighted')
print('Recall: %f' % recall)

microf1 = f1_score(true_labels, predictions, average='micro')
print('Micro F1 score: %f' % microf1)
macrof1 = f1_score(true_labels, predictions, average='macro')
print('Macro F1 score: %f' % macrof1)

target_names = ['No-Stall', 'Stall']
# Print precision-recall report
print(classification_report(true_labels, predictions, target_names=target_names))

# Save model
model_path = "/tmp/model.pt"
torch.save(model.state_dict(), model_path)
