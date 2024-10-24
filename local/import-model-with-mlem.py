from mlem.api import import_object, save
import torch
from torch import nn
import pandas as pd
import numpy as np
import torch

pd.set_option('display.max_columns', None)

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
        
path_to_pt = "../model-archiver/model-store/youtubegoes5g/model.pt"

model = InterruptionModel()
model.load_state_dict(torch.load(path_to_pt, weights_only=True))
model.eval()

# Step 1: Read the CSV file and drop the specified columns
sample = pd.read_csv('dataset.csv', nrows=1).drop(columns=['Stall', 'ID', 'Quality', 'Time'], errors='ignore')

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

save(model, "models/youtubegoes5g", sample_data=first_sample_tensor)

#model = import_object(path="../model-archiver/model-store/youtubegoes5g/model.pt", target="models/youtubegoes5g.mlem", type_="pickle")

