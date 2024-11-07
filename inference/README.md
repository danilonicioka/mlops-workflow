export INGRESS_HOST=localhost
export INGRESS_PORT=8080
MODEL_NAME=youtubegoes5g
SERVICE_HOSTNAME=$(kubectl get inferenceservice youtubegoes5g -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./youtubegoes5g-input.json