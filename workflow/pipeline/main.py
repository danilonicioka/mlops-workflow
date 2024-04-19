import kfp
from kfp import dsl

@dsl.component
def download_data(url: str, filename: str) -> str:
    from urllib.request import urlretrieve
    path, headers = urlretrieve(url, filename)
    return path

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