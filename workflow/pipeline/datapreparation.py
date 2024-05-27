import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import torch

df = pd.read_csv(local_file_path)

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

df['Stall'].replace('Yes', 1, inplace=True)
df['Stall'].replace('No', 0, inplace=True)

X = df[['CQI1', 'CQI2', 'CQI3', 'cSTD CQI',
       'cMajority', 'c25 P', 'c50 P', 'c75 P', 'RSRP1', 'RSRP2', 'RSRP3',
       'pMajority', 'p25 P', 'p50 P', 'p75 P', 'RSRQ1', 'RSRQ2', 'RSRQ3',
       'qMajority', 'q25 P', 'q50 P', 'q75 P', 'SNR1', 'SNR2', 'SNR3',
       'sMajority', 's25 P', 's50 P', 's75 P']].values

y = df['Stall'].values

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42
)