import kfp
from kfp.dsl import component, Input, Model

@component(base_image="python:3.11.9", packages_to_install=['kserve','kubernetes'])
def model_serving(
    model_trained_artifact : Input[Model]
    ):
    # Create kserve instance
    from kubernetes import client 
    from kserve import KServeClient, constants, utils, V1beta1InferenceService, V1beta1InferenceServiceSpec, V1beta1PredictorSpec, V1beta1TorchServeSpec
    from datetime import datetime
    import time
    
    #get model uri
    uri = model_trained_artifact.uri
    #replace minio with s3
    uri = uri.replace("minio","s3")
    
    #TFServing wants this type of structure ./models/1/model
    # the number represent the model version
    # in this example we use only 1 version
    
    #Inference server config
    namespace = utils.get_default_target_namespace()
    now = datetime.now()
    name="youtubegoes5g"
    kserve_version='v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    isvc = V1beta1InferenceService(api_version=api_version,
                                   kind=constants.KSERVE_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false'}),
                                   spec=V1beta1InferenceServiceSpec(
                                   predictor=V1beta1PredictorSpec(
                                       service_account_name="sa-minio-kserve",
                                       pytorch=(V1beta1TorchServeSpec(
                                           storage_uri=uri))))
    )

    KServe = KServeClient()
    
    #replace old inference service with a new one
    try:
        KServe.delete(name=name, namespace=namespace)
        print("Old model deleted")
    except:
        print("Couldn't delete old model")
    time.sleep(10)
    
    KServe.create(isvc)