#!/bin/bash

URL="http://localhost:8085/v1/models/youtubegoes5g:predict"
INPUT_FILE="input.json"
HOST="youtubegoes5g.kubeflow-user-example-com.svc.cluster.local"
CONCURRENT_REQUESTS=5

# Function to send a single request
send_request() {
    local payload=$1
    echo "Sending request with payload: $payload"
    start_time=$(date +%s%N)
    response=$(curl -s -X POST -H "Host: $HOST" -H "Content-Type: application/json" -d "$payload" $URL)
    end_time=$(date +%s%N)
    elapsed_time=$((end_time - start_time))
    echo "Response: $response"
    echo "Response Time: $((elapsed_time / 1000000)) ms"
    echo "-----------------------------------------"
}

export -f send_request
export URL HOST

# Use GNU parallel to run requests concurrently
cat "$INPUT_FILE" | parallel -j $CONCURRENT_REQUESTS send_request
