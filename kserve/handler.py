from ts.torch_handler.base_handler import BaseHandler
import torch
import os
from torch import nn
import logging
import joblib
from sklearn.preprocessing import StandardScaler

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
            
            # Load scaler
            scaler = StandardScaler()
            scaler = joblib.load('/mnt/models/model-store/youtubegoes5g/scaler.save')
            
            tensor_list = []
            for item in data:
                item = scaler.transform([item['data']])
                tensor_data = torch.tensor(item, dtype=torch.float32)  # Each instance as a tensor
                tensor_data = torch.tensor([item['data']], dtype=torch.float32)  # Each instance as a tensor
                tensor_list.append(tensor_data)
            # Stack all tensors along a new dimension to create a single tensor
            combined_tensor = torch.cat(tensor_list, dim=0)
            logger.info("Input data preprocessed successfully")
            return combined_tensor
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise ValueError("Failed to preprocess input data")

    def inference(self, model_input):
        """
        Perform model inference.
        """
        try:
            inference_list = []
            for tensor_data in model_input:
                with torch.no_grad():
                    output = torch.round(torch.sigmoid(self.model(tensor_data))).squeeze()
                inference = output.cpu().numpy().tolist()
                inference_list.append(output)
            logger.info("Inference performed successfully")
            return inference_list
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise RuntimeError("Inference failed")

    def postprocess(self, inference_output):
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
            logger.info("Output postprocessed successfully")
            return result_list
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
