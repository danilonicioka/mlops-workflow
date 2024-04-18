import kfp
import kfp.components as comp
import dvc

import subprocess
process = subprocess.Popen(['echo', 'More output'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
stdout, stderr