mlops_worflow

==============================
Local run
```
k port-forward svc/minio-svc -n minio 9090 9000
```
```
k port-forward svc/ml-pipeline-ui -n kubeflow 3000:80
```

==============================

test from End to End Machine learning pipeline with MLOps tools

Project Organization
------------

    ├── LICENSE
    ├── README.md                   <- The top-level README for developers using this project.
    ├── data
    │   ├── external                <- Data from third party sources.
    │   ├── interim                 <- Intermediate data that has been transformed.
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   └── raw                     <- The original, immutable data dump.
    |
    ├── docker                      <- files to create image for flask app
    │   ├── templates               <- html files for flask app pages
    │   │   ├── home.html           <- home page "/"
    |   |   └── submit_file.html    <- page to submit new data to append in dataset
    │   |
    │   ├── .env                    <- file to set env vars for flask app to access minio and github
    │   ├── app.py                  <- main app file
    │   ├── requirements.txt        <- packages needed to run flask app
    |
    ├── flask                       <- files to deploy flask app in cluster
    │   ├── cm.yml                  <- config map manifest
    │   ├── deployment.yml          
    │   ├── secret.yml              <- to access minio
    |
    ├── kserve                      <- files to test kserve
    ├── kubeflow                    <- additional files to kubeflow
    ├── minio                       <- manifests to deploy minio
    │
    ├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   |                              the creator's initials, and a short `-` delimited description, e.g.
    │   |                              `1.0-jqp-initial-data-exploration`.
    │   ├── 3s-DNN.ipynb            <- notebook from youtubegoes5g
    │   ├── 3sdnn.py                <- just the python script parts
    |
    ├── pipeline                    <- files to deploy kubeflow pipeline