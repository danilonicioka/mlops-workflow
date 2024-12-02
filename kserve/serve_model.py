from dotenv import load_dotenv
from kubernetes import client 
from kserve import KServeClient, constants, V1beta1InferenceService, V1beta1InferenceServiceSpec, V1beta1PredictorSpec, V1beta1TorchServeSpec
from datetime import datetime
import time

# Load environment variables from env file
load_dotenv()

# MinIO variables
MINIO_MODEL_BUCKET_NAME = "model-files"

# Model variables
MODEL_NAME = "youtubegoes5g"

# Kserve variables
KSERVE_NAMESPACE = "kubeflow-user-example-com"
KSERVE_SVC_ACC = "sa-minio-kserve"

def model_serving(
    bucket_name: str,
    model_name: str,
    kserve_namespace: str,
    kserve_svc_acc: str
):
    # Create kserve instance
        
    #Inference server config
    now = datetime.now()
    kserve_version='v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    # with open(model_uri.path) as f:
    #     uri = f.read()
    uri = f's3://{bucket_name}'

    isvc = V1beta1InferenceService(api_version=api_version,
                                    kind=constants.KSERVE_KIND,
                                    metadata=client.V1ObjectMeta(
                                        name=model_name, namespace=kserve_namespace, annotations={'sidecar.istio.io/inject':'false'}),
                                    spec=V1beta1InferenceServiceSpec(
                                    predictor=V1beta1PredictorSpec(
                                        service_account_name=kserve_svc_acc,
                                        pytorch=(V1beta1TorchServeSpec(
                                            storage_uri=uri))))
    )

    KServe = KServeClient()

    #replace old inference service with a new one
    try:
        KServe.delete(name=model_name, namespace=kserve_namespace)
        print("Old model deleted")
    except:
        print("Couldn't delete old model")
    time.sleep(10)

    KServe.create(isvc)

model_serving(MINIO_MODEL_BUCKET_NAME, MODEL_NAME, KSERVE_NAMESPACE, KSERVE_SVC_ACC)