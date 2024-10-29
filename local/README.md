export INGRESS_HOST=localhost
export INGRESS_PORT=8080

curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./youtubegoes5g-input.json