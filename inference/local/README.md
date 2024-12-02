# Create pod and pvc
k apply -f manifests

# Copy files to pod for local test

kubectl cp ../../model-archiver/model-store/youtubegoes5g local-model:/pv/youtubegoes5g
kubectl cp tests local-model:/pv/tests

# Install requirements
pip install -r pv/youtubegoes5g/requirements.txt

## To install torch on the pod if it is causing timeout
pip install torch --no-cache-dir

# Types of tests

## Single request
```
time kubectl exec local-model -- bash /pv/tests/single.py
```

## Batched - Single request, but multiple data
```
time kubectl exec local-model -- bash /pv/tests/batched.py
```

## Sequential requests - Multiple requests one after another

```
time kubectl exec local-model -- bash /pv/tests/sequential.py
```

## Concurrent request

```
time kubectl exec local-model -- bash /pv/tests/concurrent.py
```

## Stress Testing

ab -n 1000 -c 10 -p batched.json -T "application/json" http://localhost:8085/v1/models/youtubegoes5g:predict