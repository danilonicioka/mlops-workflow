# model_handler.py

"""
KServe-compatible model handler for the InterruptionModel.
"""

import kserve
import torch
from torch import nn
import numpy as np

class InterruptionModel(nn.Module):
    """
    The Interruption Model as defined in KServe setup.
    """

    def __init__(self):
        super(InterruptionModel, self).__init__()
        self.layer_1 = nn.Linear(in_features=29, out_features=200)
        self.layer_2 = nn.Linear(in_features=200, out_features=100)
        self.layer_3 = nn.Linear(in_features=100, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


class KServeModelHandler(kserve.Model):
    """
    A KServe-compatible handler that wraps the InterruptionModel.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = InterruptionModel()
        self.ready = False

    def load(self):
        """
        Loads the model weights for inference.
        """
        # For example, the model weights can be loaded from a file
        # self.model.load_state_dict(torch.load("path/to/model_weights.pt"))
        self.model.eval()
        self.ready = True

    def preprocess(self, inputs: dict) -> torch.Tensor:
        """
        Convert input JSON data to a PyTorch Tensor.
        :param inputs: JSON dictionary containing "instances" key.
        :return: torch.Tensor ready for inference
        """
        try:
            input_data = np.array(inputs["instances"]).astype(np.float32)
            tensor_data = torch.tensor(input_data)
            if len(tensor_data.shape) == 1:
                tensor_data = tensor_data.unsqueeze(0)
            return tensor_data
        except Exception as e:
            raise ValueError(f"Error during preprocessing: {str(e)}")

    def predict(self, inputs: dict) -> dict:
        """
        Make a prediction on the processed input.
        :param inputs: JSON dictionary containing "instances" key.
        :return: JSON response with predictions
        """
        try:
            model_input = self.preprocess(inputs)
            with torch.no_grad():
                model_output = self.model(model_input).numpy()
            return {"predictions": model_output.tolist()}
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

    def postprocess(self, inference_output: torch.Tensor) -> dict:
        """
        Postprocess the model output for returning results.
        :param inference_output: Raw output from model inference
        :return: Dictionary with JSON response format
        """
        return {"predictions": inference_output.tolist()}