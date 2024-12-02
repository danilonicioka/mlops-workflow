# Copy files to pod for local test

kubectl cp ../../model-archiver/model-store/youtubegoes5g local-model:/pv/youtubegoes5g
kubectl cp tests local-model:/pv/tests

# To install torch on the pod if it is causing timeout
pip install torch --no-cache-dir