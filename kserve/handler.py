import torch
from ts.torch_handler.base_handler import BaseHandler
import json
import torch.nn as nn

class InterruptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=29, out_features=200)
        self.layer_2 = nn.Linear(in_features=200, out_features=100)
        self.layer_3 = nn.Linear(in_features=100, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

class CustomModelHandler(BaseHandler):
    """
    Custom handler for TorchServe to process input features for the model
    and return predictions.
    """

    def initialize(self, context):
        """This method initializes the model and loads artifacts"""
        self.manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the model
        self.model = InterruptionModel()  # Assuming the model is defined inline
        self.model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        """
        Preprocess the input data before passing it to the model.
        The input data should be in the format:
        CQI1, CQI2, CQI3, cSTD CQI, cMajority, c25 P, c50 P, c75 P,
        RSRP1, RSRP2, RSRP3, pMajority, p25 P, p50 P, p75 P,
        RSRQ1, RSRQ2, RSRQ3, qMajority, q25 P, q50 P, q75 P,
        SNR1, SNR2, SNR3, sMajority, s25 P, s50 P, s75 P
        """
        try:
            # Extract the input data from the request
            data = data[0].get('body') if isinstance(data, list) else data.get('body')
            input_json = json.loads(data)

            # Ensure input order matches the expected feature order
            features = [
                input_json['CQI1'], input_json['CQI2'], input_json['CQI3'], input_json['cSTD CQI'],
                input_json['cMajority'], input_json['c25 P'], input_json['c50 P'], input_json['c75 P'],
                input_json['RSRP1'], input_json['RSRP2'], input_json['RSRP3'], input_json['pMajority'],
                input_json['p25 P'], input_json['p50 P'], input_json['p75 P'],
                input_json['RSRQ1'], input_json['RSRQ2'], input_json['RSRQ3'], input_json['qMajority'],
                input_json['q25 P'], input_json['q50 P'], input_json['q75 P'],
                input_json['SNR1'], input_json['SNR2'], input_json['SNR3'], input_json['sMajority'],
                input_json['s25 P'], input_json['s50 P'], input_json['s75 P']
            ]

            # Convert input data to tensor and reshape as necessary
            input_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)

        except Exception as e:
            raise ValueError(f"Error during preprocessing: {str(e)}")

        return input_tensor

    def inference(self, data):
        """
        Perform inference on the preprocessed data and return the prediction.
        """
        with torch.no_grad():
            output = self.model(data)
        return output

    def postprocess(self, inference_output):
        """
        Post-process the model output to return a response.
        """
        try:
            # Convert the output tensor to a JSON-compatible format
            result = inference_output.cpu().numpy().tolist()
            return [result]

        except Exception as e:
            raise ValueError(f"Error during postprocessing: {str(e)}")

    def handle(self, data, context):
        """
        Main handle function for TorchServe.
        """
        preprocessed_data = self.preprocess(data)
        inference_output = self.inference(preprocessed_data)
        return self.postprocess(inference_output)
