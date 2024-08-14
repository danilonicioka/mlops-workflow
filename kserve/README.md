torch-model-archiver --model-name youtubegoes5g --version 1.0 --model-file model.py --serialized-file model_trained_artifact.pt --handler base_handler --requirements-file requirements.txt

kubectl create namespace kserve-test

kubectl config set-context --current --namespace=kserve-test

kubectl apply -f pvc.yml -n kserve-test

kubectl apply -f pv_pod.yml -n kserve-test

# Create directory in PV
kubectl exec -it model-store-pod -c model-store -n kserve-test -- mkdir /pv/model-store/
kubectl exec -it model-store-pod -c model-store -n kserve-test -- mkdir /pv/config/
# Copy files the path
kubectl cp youtubegoes5g.mar model-store-pod:/pv/model-store/ -c model-store -n kserve-test
kubectl cp config.properties model-store-pod:/pv/config/ -c model-store -n kserve-test

k apply -f is_kserve.yml -n kserve-test