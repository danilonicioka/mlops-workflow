from ts.torch_handler.base_handler import BaseHandler
import torch
import os
from torch import nn

class ModelHandler(BaseHandler):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self._context = None
        self.initialized = False
        self.model = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time.
        """
        self._context = context

        #  Load the model
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Check model file exists and load model state
        serialized_file = context.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError(f"Missing the model file: {model_pt_path}")

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

        # Initialize and load the model
        self.model = InterruptionModel().to(self.device)
        self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        """
        # Parse and check data
        input_data = data[0].get("data") or data[0].get("body")
        if input_data is None or "instances" not in input_data:
            raise ValueError("Invalid input format, expecting 'data' or 'body' key with 'instances'.")

        # Convert input to tensor
        tensor_data = torch.tensor(input_data["instances"], dtype=torch.float32).to(self.device)
        return tensor_data

    def inference(self, model_input):
        """
        Perform model inference.
        """
        with torch.no_grad():
            output = self.model(model_input)
        return output

    def postprocess(self, inference_output):
        """
        Convert model output to a list of predictions.
        """
        return inference_output.cpu().numpy().tolist()

    def handle(self, data, context):
        """
        Handle a prediction request.
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
