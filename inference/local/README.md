# Copy files to pod for local test

kubectl cp ../kserve/model.pt local-model:/pv/
kubectl cp ../notebooks/scaler.pkl local-model:/pv/
kubectl cp test.py local-model:/pv/
kubectl cp ../kserve/requirements.txt local-model:/pv/

pip install torch --no-cache-dir