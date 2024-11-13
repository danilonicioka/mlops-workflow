from ts.torch_handler.base_handler import BaseHandler
import torch
import os
from torch import nn
import logging

logger = logging.getLogger(__name__)

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
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
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
        
        try:
            # Initialize and load the model
            self.model = InterruptionModel().to(self.device)
            self.model.load_state_dict(torch.load(model_pt_path, map_location=self.device))
            self.model.eval()
            self.initialized = True
            logger.info("Model loaded and ready for inference")
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            raise RuntimeError("Model initialization failed")

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        """
        try:
            # Log the incoming data for debugging
            logger.info(f"Received data: {data}")

            # Access all instances for batch processing
            instances = [item.get("data") for item in data if "data" in item]

            if not instances:
                raise ValueError("No 'data' field found in the input.")

            # Convert to tensor for batch processing
            tensor_data = torch.tensor(instances, dtype=torch.float32).to(self.device)
            logger.info("Input data preprocessed successfully")
            return tensor_data
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise ValueError("Failed to preprocess input data")

    def inference(self, model_input):
        """
        Perform model inference.
        """
        try:
            with torch.no_grad():
                # Use sigmoid for binary classification, rounding to 0 or 1
                output = torch.round(torch.sigmoid(self.model(model_input))).squeeze()
            logger.info("Inference performed successfully")
            return output
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise RuntimeError("Inference failed")

    def postprocess(self, inference_output):
        """
        Convert model output to a list of predictions.
        """
        try:
            # Process each item in the batch
            predictions = inference_output.cpu().numpy()
            results = ["Stall" if pred > 0 else "No Stall" for pred in predictions]
            logger.info("Output postprocessed successfully")
            return results
        except Exception as e:
            logger.error(f"Error during postprocessing: {str(e)}")
            raise ValueError("Failed to postprocess output data")

    def handle(self, data, context):
        """
        Handle a prediction request.
        """
        try:
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            return self.postprocess(model_output)
        except Exception as e:
            logger.error(f"Error during handle: {str(e)}")
            return [str(e)]
