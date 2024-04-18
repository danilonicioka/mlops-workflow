import kfp
import kfp.components as comp
import kfp.dsl as dsl
import dvc

# # merge func
# def merge_csv(file_path: comp.InputPath('Tarball'),
#               output_csv: comp.OutputPath('CSV')):
#   import glob
#   import pandas as pd
#   import tarfile

#   tarfile.open(name=file_path, mode="r|gz").extractall('data')
#   df = pd.concat(
#       [pd.read_csv(csv_file, header=None)
#        for csv_file in glob.glob('data/*.csv')])
#   df.to_csv(output_csv, index=False, header=False)

# # create kfp component from merge func
# create_step_merge_csv = kfp.components.create_component_from_func(
#     func=merge_csv,
#     output_component_file='component.yaml', # This is optional. It saves the component spec for future use.
#     base_image='python:3.8',
#     packages_to_install=['pandas==1.1.4'])

# load download component from github
web_downloader_op = kfp.components.load_component_from_url(
    'https://raw.githubusercontent.com/kubeflow/pipelines/3fa2ac5f4e04111bf5758fd5c01a2f0d7ac4b866/components/contrib/web/Download/component.yaml')

@dsl.component
# Define a pipeline and create a task from a component:
def my_pipeline(url: any):
  web_downloader_task = web_downloader_op(url=url)
#   merge_csv_task = create_step_merge_csv(file=web_downloader_task.outputs['data'])
  # The outputs of the merge_csv_task can be referenced using the
  # merge_csv_task.outputs dictionary: merge_csv_task.outputs['output_csv']

kfp.compiler.Compiler().compile(
    pipeline_func=my_pipeline,
    package_path='pipeline.yaml')

client = kfp.Client(host="http://localhost:3000")

print(client.list_experiments())

client.create_run_from_pipeline_func(
    my_pipeline,
    arguments={
        'url': 'https://github.com/razaulmustafa852/youtubegoes5g/blob/main/Models/Stall-Windows%20-%20Stall-3s.csv'
    })