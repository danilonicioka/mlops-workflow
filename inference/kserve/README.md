# Inference Request

export INGRESS_HOST=localhost
export INGRESS_PORT=8085
MODEL_NAME=youtubegoes5g
SERVICE_HOSTNAME=$(kubectl get inferenceservice youtubegoes5g -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./youtubegoes5g-input.json -w "\nResponse Time: %{time_total}s\n" 

# Concurrent request



# Stress Testing

ab -n 1000 -c 10 -p batched.json -T "application/json" http://localhost:8085/v1/models/youtubegoes5g:predict