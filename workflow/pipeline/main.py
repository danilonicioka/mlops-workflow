import kfp
from kfp import dsl

@dsl.component(packages_to_install=['Minio','cryptography','Werkzeug'])
def download_data(url: str, filename: str) -> str:
    from urllib.request import urlretrieve
    from werkzeug.utils import secure_filename
    from minio import Minio
    import os
    def upload_file(path):
        with open(path) as f: data = f.read()
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if path == "":
            return "file not found"
        if path:
            filename = secure_filename(path)
            size = os.stat(path).st_size
            msg = upload_object(filename, data, size)
            return msg
    def upload_object(filename, data, length):
        class Dataset:
            def __init__(self, data=None):
                # Initialize the data attribute with the provided data or an empty list by default
                self.data = data if data is not None else []
        dataset = Dataset(data=data)
        client = Minio(MINIO_ENDPOINT, MINIO_USER, MINIO_PASS, secure=False)
        # Make bucket if not exist.
        found = client.bucket_exists(BUCKET_NAME)
        if not found:
            client.make_bucket(BUCKET_NAME)
        client.put_object(BUCKET_NAME, filename, dataset, length)
        return f"{filename} is successfully uploaded to bucket {BUCKET_NAME}."
    # download file
    path, headers = urlretrieve(url, filename)
    # minio credentials
    MINIO_USER = "minioadmin"
    MINIO_PASS = "minioadmin"
    BUCKET_NAME = "flask-minio"
    MINIO_ENDPOINT = "minio-svc.minio:9000"
    msg = upload_file(path)
    return msg

@dsl.pipeline
def my_pipeline(url: str, filename: str) -> str:
    download_task = download_data(url=url, filename=filename)
    return download_task.output

kfp.compiler.Compiler().compile(
    pipeline_func=my_pipeline,
    package_path='pipeline.yaml')

url = "https://github.com/razaulmustafa852/youtubegoes5g/blob/main/Models/Stall-Windows%20-%20Stall-3s.csv"
filename = "init_dataset.csv"

client = kfp.Client(host="http://localhost:3000")

client.create_run_from_pipeline_func(
    my_pipeline,
    arguments={
        'url': url,
        'filename': filename
    })
