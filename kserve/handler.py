import io
from ts.torch_handler.base_handler import BaseHandler
import torch
import json
import os
import numpy as np

class InterruptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=29, out_features=200)
        self.layer_2 = nn.Linear(in_features=200, out_features=100)
        self.layer_3 = nn.Linear(in_features=100, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

class InterruptionModelHandler(BaseHandler):
    def __init__(self):
        super(InterruptionModelHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        """
        Initialize the model. This function is called during model loading.
        """
        # Load the model
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        model_file = os.path.join(model_dir, "model.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved model state
        self.model = InterruptionModel()
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True

    def preprocess(self, requests):
        """
        Preprocess the requests and return batched inputs to the model.
        Input features are expected to be in JSON format.
        """
        all_inputs = []
        for req in requests:
            data = req.get("data") or req.get("body")
            # Convert data to float and form a tensor
            features = json.loads(data)
            input_tensor = torch.tensor([features], dtype=torch.float32, device=self.device)
            all_inputs.append(input_tensor)
        
        # Batch the inputs
        inputs = torch.cat(all_inputs, dim=0)
        return inputs

    def inference(self, inputs):
        """
        Run the model inference and return the output.
        """
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

    def postprocess(self, inference_outputs):
        """
        Convert the model output to a final response format.
        """
        # In this case, the model outputs a single value for each input, so we return it as a list.
        return inference_outputs.squeeze().cpu().numpy().tolist()

    def handle(self, data, context):
        """
        Entry point for TorchServe.
        """
        if not self.initialized:
            self.initialize(context)
        
        # Preprocess the request input
        inputs = self.preprocess(data)
        
        # Run inference on the preprocessed data
        outputs = self.inference(inputs)
        
        # Post-process and return the result
        result = self.postprocess(outputs)
        return [result]

