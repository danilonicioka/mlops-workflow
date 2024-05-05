import os
from typing import Tuple
from kfp.components import create_component_from_func
from kfp import dsl
from minio import Minio
from minio.error import S3Error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Global constants
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_BUCKET = "dvc-data"
DATA_DIR = "/data"
DATA_PATH = os.path.join(DATA_DIR, "init_dataset.csv")
RANDOM_STATE = 42

# Define common paths and constants
TRAINING_PATH = os.path.join(DATA_DIR, "training_data.npz")
TESTING_PATH = os.path.join(DATA_DIR, "testing_data.npz")

# Define experiment and pipeline names
EXPERIMENT_NAME_PREPROCESS = "DataPreprocessingExperiment"
PIPELINE_NAME_PREPROCESS = "DataPreprocessingPipeline"
EXPERIMENT_NAME_MINIO = "RetrieveDataFromMinioExperiment"
PIPELINE_NAME_MINIO = "RetrieveDataFromMinioPipeline"
OBJECT_NAME = "init_dataset.csv"

# Data ingestion function
def retrieve_data_from_minio(object_name: str) -> None:
    """
    Retrieve data from Minio database and save it to a predefined file path.

    Parameters:
    - object_name: The name of the object in the Minio bucket.
    """
    # Retrieve Minio credentials from environment variables
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")

    # Check if Minio credentials are provided
    if not access_key or not secret_key:
        logging.error("Minio credentials (access key and secret key) must be provided via environment variables.")
        raise ValueError("Minio credentials (access key and secret key) must be provided via environment variables.")

    # Initialize the Minio client
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=access_key,
        secret_key=secret_key,
        secure=False  # Set to True if using HTTPS
    )

    # Retrieve the object from the specified bucket and save it to the file path
    try:
        client.fget_object(MINIO_BUCKET, object_name, DATA_PATH)
        logging.info(f"Data retrieved from Minio and saved to {DATA_PATH}.")
    except S3Error as e:
        logging.error(f"Failed to retrieve data from Minio: {e}")
        raise

# Create the KFP component from the function
retrieve_data_from_minio_component = create_component_from_func(
    retrieve_data_from_minio,
    base_image="python:3.9",
    packages_to_install=["minio"],
    output_component_file="retrieve_data_from_minio_component.yaml"
)

## Data preparation function
def preprocess_data(
    data_path: str = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[str, str]:
    """
    Load data, preprocess it, and split it into training and testing sets.

    Parameters:
    - data_path: Path to the CSV file containing the data.
    - test_size: Proportion of the data to use as the test set.
    - random_state: Seed for random number generator.
    
    Returns:
    - Tuple of file paths to the training data and testing data.
    """
    try:
        # Load the data
        df = pd.read_csv(data_path)

        # Columns to convert to numeric types and handle missing values
        columns_to_convert = [
            'CQI1', 'CQI2', 'CQI3', 'cSTD CQI', 'cMajority', 'c25 P', 'c50 P', 'c75 P',
            'RSRP1', 'RSRP2', 'RSRP3', 'pMajority', 'p25 P', 'p50 P', 'p75 P',
            'RSRQ1', 'RSRQ2', 'RSRQ3', 'qMajority', 'q25 P', 'q50 P', 'q75 P',
            'SNR1', 'SNR2', 'SNR3', 'sMajority', 's25 P', 's50 P', 's75 P'
        ]

        # Convert selected columns to float type and handle missing values
        df[columns_to_convert] = df[columns_to_convert].astype(float)
        df[columns_to_convert] = df[columns_to_convert].fillna(df[columns_to_convert].mean())

        # Replace 'Yes' and 'No' in 'Stall' column with 1 and 0
        df['Stall'].replace({'Yes': 1, 'No': 0}, inplace=True)

        # Extract features (X) and target (y)
        X = df[columns_to_convert].values
        y = df['Stall'].values

        # Resample the dataset to balance classes using SMOTE
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert X and y to PyTorch tensors
        X = torch.from_numpy(X).type(torch.float32)
        y = torch.from_numpy(y).type(torch.float32)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Save the training and testing sets to files
        np.savez_compressed(TRAINING_PATH, X_train=X_train.numpy(), y_train=y_train.numpy())
        np.savez_compressed(TESTING_PATH, X_test=X_test.numpy(), y_test=y_test.numpy())

        logging.info(f"Data preprocessed and saved to {TRAINING_PATH} and {TESTING_PATH}.")
        return TRAINING_PATH, TESTING_PATH
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise

# Create the KFP component from the function
preprocess_data_component = create_component_from_func(
    preprocess_data,
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "torch", "scikit-learn", "imblearn"],
    output_component_file="preprocess_data_component.yaml"
)

## Define the pipelines
@dsl.pipeline(
    name="RetrieveDataFromMinioPipeline",
    description="A pipeline to retrieve data from a Minio database."
)
def retrieve_data_pipeline(object_name: str):
    # Create a task using the retrieve data from Minio component
    retrieve_data_task = retrieve_data_from_minio_component(object_name=object_name)

# Compile the pipeline to a YAML file
pipeline_filename = 'retrieve_data_pipeline.yaml'
kfp.compiler.Compiler().compile(pipeline=retrieve_data_pipeline, package_path=pipeline_filename)

# Define a data preprocessing pipeline using KFP DSL
@dsl.pipeline(
    name="DataPreprocessingPipeline",
    description="A pipeline to preprocess data and split it into training and testing sets."
)
def data_preprocessing_pipeline():
    # Create a task using the preprocessing data component
    preprocess_data_task = preprocess_data_component()

# Compile the data preprocessing pipeline to a YAML file
pipeline_filename = 'data_preprocessing_pipeline.yaml'
kfp.compiler.Compiler().compile(pipeline=data_preprocessing_pipeline, package_path=pipeline_filename)

# Run the pipelines
def run_pipelines():
    # Specify the KFP endpoint
    kfp_endpoint = "http://localhost:3000"

    # Create the KFP client
    client = kfp.Client(host=kfp_endpoint)

    # Run the data preprocessing pipeline
    experiment = client.get_experiment(EXPERIMENT_NAME_PREPROCESS)
    if experiment is None:
        experiment = client.create_experiment(EXPERIMENT_NAME_PREPROCESS)

    run = client.create_run_from_pipeline_package(
        package_path=pipeline_filename,
        experiment_id=experiment.experiment_id,
        run_name="DataPreprocessingRun"
    )
    logging.info(f"Data preprocessing pipeline run created with ID: {run.run_id}")

    # Run the retrieve data pipeline
    experiment = client.get_experiment(EXPERIMENT_NAME_MINIO)
    if experiment is None:
        experiment = client.create_experiment(EXPERIMENT_NAME_MINIO)

    run = client.create_run_from_pipeline_package(
        package_path=pipeline_filename,
        experiment_id=experiment.experiment_id,
        run_name="RetrieveDataFromMinioRun",
        pipeline_parameters={"object_name": OBJECT_NAME},
    )

    logging.info(f"Retrieve data pipeline run created with ID: {run.run_id}")

# Run the pipelines
if __name__ == "__main__":
    run_pipelines()
