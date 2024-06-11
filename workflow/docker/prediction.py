import numpy as np
from kserve import utils
import requests

def send_request(x_array):
'''
This function takes a numpy array as input parameter and print the class id
'''
  # endpoint setup
  name="digits-recognizer"
  namespace = utils.get_default_target_namespace()
  
  url="http://{}.{}.svc.cluster.local/v1/models/{}:predict".format(name,namespace,name)
  
  # array to string operation
  data_formatted = np.array2string(x_array, separator=",", formatter={"float": lambda x: "%.1f" % x})
  # string to json
  json_request = '{{ "instances" : {} }}'.format(data_formatted)

  #sending request
  response = requests.post(url, data=json_request)

  # reading response
  json_response = response.json()
  print("Predicted: {}".format(np.argmax(json_response["predictions"])))