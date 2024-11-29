#!/bin/bash

URL="http://localhost:8085/v1/models/youtubegoes5g:predict"
BATCH_FILE="inputs.json"
HOST=youtubegoes5g.kubeflow-user-example-com.svc.cluster.local

start_time=$(date +%s%N)
response=$(curl -s -X POST -H "Host: $HOST" -H "Content-Type: application/json" -d @"$BATCH_FILE" $URL)
end_time=$(date +%s%N)
elapsed_time=$((end_time - start_time))

echo "Response: $response"
echo "Response Time: $((elapsed_time / 1000000)) ms"
