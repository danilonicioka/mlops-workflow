from ts.torch_handler.base_handler import BaseHandler
import torch

class ModelHandler(BaseHandler):
    def initialize(self, context):
        # Load the model definition from model.py and set the weights from model.pt
        model_dir = context.system_properties.get("model_dir")
        serialized_file = context.system_properties.get("serialized_file")

        # Load model architecture and weights
        self.model = torch.jit.load(f"{model_dir}/{serialized_file}")
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        # Preprocess logic
        data = data[0].get("data") or data[0].get("body")
        tensor_data = torch.tensor(data["instances"], dtype=torch.float32)
        return tensor_data

    def inference(self, model_input):
        with torch.no_grad():
            return self.model(model_input)

    def postprocess(self, inference_output):
        return inference_output.tolist()
