from mlem.api import load
import pandas as pd
import torch
import numpy as np

model = load("/home/danilo/tcc/mlops-workflow/local/models/youtubegoes5g")  # RandomForestClassifier

# Step 1: Read the CSV file and drop the specified columns
sample = pd.read_csv('/home/danilo/tcc/mlops-workflow/data/external/dataset.csv', nrows=1).drop(columns=['Stall', 'ID', 'Quality', 'Time'], errors='ignore')

# Step 2: Replace ' ', '-', and np.nan with 0
sample = sample.replace([' ', '-', np.nan], 0)

# Convert all columns to float 
sample = sample.astype(float)

# Step 3: Extract the first sample as a NumPy array without the column names
first_sample = sample.values#.flatten()  # Use flatten() to get a 1D array

# Step 4: Convert the first sample to a PyTorch tensor
first_sample_tensor = torch.tensor(first_sample, dtype=torch.float32)

# Display the tensor
print("First Sample Tensor:", first_sample_tensor)

y_pred = model.forward(first_sample_tensor)

print(y_pred)