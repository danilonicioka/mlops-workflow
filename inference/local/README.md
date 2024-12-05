# Create pod and pvc
k apply -f manifests

# Copy files to pod for local test

kubectl cp tests local-model:/pv
kubectl cp ../../model-archiver/model-store/youtubegoes5g/model.py local-model:/pv/tests/model.py
kubectl cp ../../model-archiver/model-store/youtubegoes5g/model.pt local-model:/pv/tests/model.pt
kubectl cp ../../model-archiver/model-store/youtubegoes5g/scaler.pkl local-model:/pv/tests/scaler.pkl
kubectl cp ../../model-archiver/model-store/youtubegoes5g/requirements.txt local-model:/pv/tests/requirements.txt

# Install requirements
pip install -r pv/youtubegoes5g/requirements.txt

## To install torch on the pod if it is causing timeout
pip install torch --no-cache-dir

# Types of tests

## Single request
```
time kubectl exec local-model -- python /pv/tests/single.py
```

## Batched - Single request, but multiple data
```
time kubectl exec local-model -- python /pv/tests/batched.py
```

## Sequential requests - Multiple requests one after another

```
time kubectl exec local-model -- python /pv/tests/sequential.py
```

## Concurrent request

```
time kubectl exec local-model -- python /pv/tests/concurrent_n.py
```

## Stress Testing

