# Commands to test kserve manually

torch-model-archiver -f --model-name youtubegoes5g --version 1.0 --model-file ../model-archiver/model-store/youtubegoes5g/model.py --serialized-file model.pt --handler handler.py -r requirements.txt --extra-files scaler.save -c ../model-archiver/config.properties

kubectl create namespace kserve-test

kubectl config set-context --current --namespace=kserve-test

kubectl apply -f pvc.yml

kubectl apply -f pv_pod.yml

# Create directory in PV
kubectl exec -it model-store-pod -c model-store -- mkdir /pv/model-store/
kubectl exec -it model-store-pod -c model-store -- mkdir /pv/config/
# Copy files the path
kubectl cp youtubegoes5g.mar model-store-pod:/pv/model-store/ -c model-store
kubectl cp ../model-archiver/config/config.properties model-store-pod:/pv/config/ -c model-store

k apply -f is.yml -n kserve-test