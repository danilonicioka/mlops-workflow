import torch
import os
from torch import nn
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess(data):
    """
    Transform raw input into model input data.
    """
    try:
        # Log the incoming data for debugging
        logger.info(f"Received data: {data}")

        # Load scaler
        scaler = StandardScaler()
        scaler = joblib.load('scaler2.save')

        tensor_list = []
        for item in data:
            item = scaler.transform([item['data']])
            tensor_data = torch.tensor(item, dtype=torch.float32)  # Each instance as a tensor
            tensor_list.append(tensor_data)
        # Stack all tensors along a new dimension to create a single tensor
        combined_tensor = torch.cat(tensor_list, dim=0)
        logger.info("Input data preprocessed successfully")
        return combined_tensor
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise ValueError("Failed to preprocess input data")

def inference(model_input):
    """
    Perform model inference.
    """
    try:
        inference_list = []
        for tensor_data in model_input:
            with torch.no_grad():
                output = torch.round(torch.sigmoid(model_3(tensor_data))).squeeze()
            inference = output.cpu().numpy().tolist()
            inference_list.append(output)
        logger.info("Inference performed successfully")
        return inference_list
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise RuntimeError("Inference failed")

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
        logger.info("Output postprocessed successfully")
        return result_list
    except Exception as e:
        logger.error(f"Error during postprocessing: {str(e)}")
        raise ValueError("Failed to postprocess output data")

def handle(data):
    """
    Handle a prediction request.
    """
    try:
        model_input = preprocess(data)
        model_output = inference(model_input)
        return postprocess(model_output)
    except Exception as e:
        logger.error(f"Error during handle: {str(e)}")
        return [str(e)]
    
no_data = [{'data': [13,13,13,0,13,13,13,13,-76,-76,-81,-76,-78.5,-76,-76,-7,-7,-12,-7,-9.5,-7,-7,12,12,7,12,9.5,12,12]}]
stall_data = [{'data': [14,14,14,0,14,14,14,14,-99,-99,-99,-99,-99,-99,-99,-5,-10,-10,-10,-10,-10,-7.5,17,17,17,17,17,17,17]}]
m_data = [{'data': [13,13,13,0,13,13,13,13,-76,-76,-81,-76,-78.5,-76,-76,-7,-7,-12,-7,-9.5,-7,-7,12,12,7,12,9.5,12,12]}, {'data': [14,14,14,0,14,14,14,14,-99,-99,-99,-99,-99,-99,-99,-5,-10,-10,-10,-10,-10,-7.5,17,17,17,17,17,17,17]}]

handle(m_data)