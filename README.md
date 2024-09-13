Project Organization
------------

    ├── data
    │   ├── external                            <- Data from third party sources.
    │   ├── interim                             <- Intermediate data that has been transformed.
    │   ├── processed                           <- The final, canonical data sets for modeling.
    │   └── raw                                 <- The original, immutable data dump.
    │
    ├── flask                                   <- files to deploy flask app in cluster
    │   ├── docker                              <- files to create image for flask app
    │   │   ├── templates                       <- html files for flask app pages
    │   │   │    ├── home.html                  <- home page "/"
    │   │   │    └── submit_file.html           <- page to submit new data to append in dataset
    │   │   ├── .env                            <- file to set env vars for flask app to access minio and github
    │   │   ├── app.py                          <- main app file
    │   │   ├── Dockerfile                      <- to build flask app image to run on k8s
    │   │   ├── prediction.py                   <- file to request prediction to model (not tested yet)
    │   │   └── requirements.txt                <- packages needed to run flask app
    │   └── k8s                                 <- files to create image for flask app
    │       ├── cm.yml                          <- config map manifest
    │       ├── deployment.yml  
    │       └── secret.yml                      <- to access minio
    │
    ├── kserve                                  <- files to test kserve
    │   ├── config.properties                   <- config for inverence service (is)
    │   ├── handler.py                          <- handler for is
    │   ├── is.yml                              <- is manifest
    │   ├── kserver-minio-secret.yaml           <- secret for kubeflow to access minio
    │   ├── model_trained_artifact.pt           <- The final, canonical data sets for modeling.
    │   ├── model.py                            <- model used in pipeline for is
    │   ├── pv_pod.yml                          <- pod to store files for is to access
    │   ├── pvc.yml                             <- pvc for pod to save files for is
    │   ├── README.md                           <- commands to run is on kserve manually
    │   └── youtubegoes5g.mar                   <- .mar file for is
    │
    ├── kubeflow                                <- additional files to kubeflow
    │   └── profile.yml                         <- manifest to give kubeflow notebooks access to run pipelines
    │
    ├── notebooks                               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   │                                      the creator's initials, and a short `-` delimited description, e.g.
    │   │                                      `1.0-jqp-initial-data-exploration`.
    │   ├── 1.0-drn-mlops-pipeline.ipynb        <- mlops pipeline notebook
    │   ├── 1.1-drn-test-components.ipynb       <- notebook to test the components in mlops pipeline notebook
    │   ├── 2.0-3s-DNN.ipynb                    <- notebook from youtubegoes5g
    │   ├── 2.1-3s-DNN-custom.ipynb             <- notebook from youtubegoes5g but edited to download database etc
    │   ├── env                                 <- env file
    │   ├── mlops.yml                           <- yaml file created by the mlops pipeline
    │   ├── requirements-components-test.txt    <- requirements to run test components notebook
    │   └── requirements.txt                    <- requirements to run mlops pipeline
    │
    ├── pipeline                                <- files to deploy kubeflow pipeline
    ├── LICENSE
    └── README.md