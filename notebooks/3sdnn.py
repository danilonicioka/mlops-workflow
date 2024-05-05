
import torch
import numpy as np


from torch import nn


torch.__version__


import pandas as pd


df = pd.read_csv('Stall-Windows - Stall-3s.csv')


df.columns


#df = df.replace([' ', '-',np.nan], 0) # There are null values
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

# Display the modified DataFrame
# print(df)


# Check which columns contain np.nan values
columns_with_nan = df.isna().any()
# Display the columns with np.nan values
# print(columns_with_nan)


df['Stall'].replace('Yes', 1, inplace=True)
df['Stall'].replace('No', 0, inplace=True)


df.columns


X = df[['CQI1', 'CQI2', 'CQI3', 'cSTD CQI',
       'cMajority', 'c25 P', 'c50 P', 'c75 P', 'RSRP1', 'RSRP2', 'RSRP3',
       'pMajority', 'p25 P', 'p50 P', 'p75 P', 'RSRQ1', 'RSRQ2', 'RSRQ3',
       'qMajority', 'q25 P', 'q50 P', 'q75 P', 'SNR1', 'SNR2', 'SNR3',
       'sMajority', 's25 P', 's50 P', 's75 P']].values


y =df['Stall'].values


X.shape, y.shape


import numpy as np


from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X.shape


X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)


X.shape


X[0]


y.shape


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42
)

X_train[:1],X_train[:1].shape, y_train[:1]


device = "cuda" if torch.cuda.is_available() else "cpu"
device


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

model_3 = InteruptionModel().to(device)
print(model_3)


# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_3.parameters(), lr=0.0001)


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
    y_logits = model_3(X_train).squeeze()

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
    model_3.eval()
    with torch.no_grad():
      # 1. Forward pass
        test_logits = model_3(X_test).squeeze()
        #print(test_logits.shape)
        test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
        # 2. Calcuate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)


    # Print out what's happening
    if epoch % 500 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")


model_3.eval()
with torch.no_grad():
     y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()


y_preds.shape,y_test.shape


predictions = y_preds.cpu().numpy() #if it is cuda, then this, otherwise y_pred.numpy()
true_labels = y_test.cpu().numpy()


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


#Done


