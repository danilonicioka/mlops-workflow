from urllib.request import urlretrieve

url = "https://raw.githubusercontent.com/razaulmustafa852/youtubegoes5g/main/Models/Stall-Windows%20-%20Stall-3s.csv"
filename = "init_dataset.csv"

with open('/home/danilo/Downloads/init_dataset.csv') as f: minio_file = f.read()

# download file
path, headers = urlretrieve(url, filename)

with open(path) as f: data = f.read()

print(data == minio_file)