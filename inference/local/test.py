import torch
import os
from torch import nn
from pickle import load
from sklearn.preprocessing import StandardScaler

def initialize():
    """
    Initialize model. This will be called during model loading time.
    """
    device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

    serialized_file = "/pv/model.pt"

    if not os.path.isfile(serialized_file):
        raise RuntimeError(f"Missing the model file: {serialized_file}")

    # Define model architecture
    class InterruptionModel(nn.Module):
        def __init__(self):
            super(InterruptionModel, self).__init__()
            self.layer_1 = nn.Linear(in_features=29, out_features=200)
            self.layer_2 = nn.Linear(in_features=200, out_features=100)
            self.layer_3 = nn.Linear(in_features=100, out_features=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

    try:
        # Initialize and load the model
        model = InterruptionModel().to(device)
        model.load_state_dict(torch.load(serialized_file, weights_only=True, map_location=device))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError("Model initialization failed:", e)

def preprocess(data):
    """
    Transform raw input into model input data.
    """
    try:
        # Load scaler
        scaler = StandardScaler()
        scaler = load(open('/pv/scaler.pkl', 'rb'))

        tensor_list = []
        for item in data:
            item = scaler.transform([item['data']])
            tensor_data = torch.tensor(item, dtype=torch.float32)  # Each instance as a tensor
            tensor_list.append(tensor_data)
        # Stack all tensors along a new dimension to create a single tensor
        combined_tensor = torch.cat(tensor_list, dim=0)
        return combined_tensor
    except Exception as e:
        raise ValueError("Failed to preprocess input data")

def inference(model_input, model):
    """
    Perform model inference.
    """
    try:
        inference_list = []
        for tensor_data in model_input:
            with torch.no_grad():
                output = torch.round(torch.sigmoid(model(tensor_data))).squeeze()
            inference = output.cpu().numpy().tolist()
            inference_list.append(output)
        return inference_list
    except Exception as e:
        raise RuntimeError("Inference failed:", e)

def postprocess(inference_output):
    """
    Convert model output to a list of predictions.
    """
    try:
        # Process each item in the batch
        result_list = []
        for result in inference_output:
            if result > 0:
                result_list.append("Stall")
            else:
                result_list.append("No Stall")
        return result_list
    except Exception as e:
        raise ValueError("Failed to postprocess output data")

def handle(data):
    """
    Handle a prediction request.
    """
    try:
        model = initialize()
        model_input = preprocess(data)
        model_output = inference(model_input, model)
        return postprocess(model_output)
    except Exception as e:
        return [str(e)]
    
no_data = [{'data': [13,13,13,0,13,13,13,13,-76,-76,-81,-76,-78.5,-76,-76,-7,-7,-12,-7,-9.5,-7,-7,12,12,7,12,9.5,12,12]}]
stall_data = [{'data': [14,14,14,0,14,14,14,14,-99,-99,-99,-99,-99,-99,-99,-5,-10,-10,-10,-10,-10,-7.5,17,17,17,17,17,17,17]}]
m_data = [{'data': [13,13,13,0,13,13,13,13,-76,-76,-81,-76,-78.5,-76,-76,-7,-7,-12,-7,-9.5,-7,-7,12,12,7,12,9.5,12,12]}, {'data': [14,14,14,0,14,14,14,14,-99,-99,-99,-99,-99,-99,-99,-5,-10,-10,-10,-10,-10,-7.5,17,17,17,17,17,17,17]}]

result = handle(m_data)
print(result)