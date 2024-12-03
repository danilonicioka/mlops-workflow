import torch
import os
from pickle import load
from sklearn.preprocessing import StandardScaler
from model import InterruptionModel

class MyHandler():
    def __init__(self):
        self.model = None
        self.device = None

    def initialize(self):
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        serialized_file = "/pv/tests/model.pt"

        if not os.path.isfile(serialized_file):
            raise RuntimeError(f"Missing the model file: {serialized_file}")

        try:
            # Initialize and load the model
            self.model = InterruptionModel().to(self.device)
            self.model.load_state_dict(torch.load(serialized_file, weights_only=True, map_location=self.device))
            self.model.eval()
        except Exception as e:
            raise RuntimeError("Model initialization failed:", e)

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        """
        try:
            # Load scaler
            scaler = StandardScaler()
            scaler = load(open('/pv/tests/scaler.pkl', 'rb'))

            tensor_list = []
            for item in data:
                item = scaler.transform([item['data']])
                tensor_data = torch.tensor(item, dtype=torch.float32)  # Each instance as a tensor
                tensor_list.append(tensor_data)
            # Stack all tensors along a new dimension to create a single tensor
            combined_tensor = torch.cat(tensor_list, dim=0)
            return combined_tensor
        except Exception as e:
            raise ValueError("Failed to preprocess input data: ", e, "data received: ", data)

    def inference(self, model_input):
        """
        Perform model inference.
        """
        try:
            inference_list = []
            for tensor_data in model_input:
                with torch.no_grad():
                    output = torch.round(torch.sigmoid(self.model(tensor_data))).squeeze()
                #inference = output.cpu().numpy().tolist()
                inference_list.append(output)
            return inference_list
        except Exception as e:
            raise RuntimeError("Inference failed:", e)

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
            return result_list
        except Exception as e:
            raise ValueError("Failed to postprocess output data: ", e)

    def handle(self, data):
        """
        Handle a prediction request.
        """
        try:
            self.initialize()
            model_input = self.preprocess(data)
            model_output = self.inference(model_input)
            return self.postprocess(model_output)
        except Exception as e:
            return [str(e)]