#!/bin/bash

URL="http://localhost:8085/v1/models/youtubegoes5g:predict"
INPUT_FILE="inputs.json"
HOST="youtubegoes5g.kubeflow-user-example-com.svc.cluster.local"
CONCURRENT_REQUESTS=4

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

# Extract "data" values from input.json and prepare payloads
total_start_time=$(date +%s%N)
jq -c '.instances[] | {instances: [.]}' "$INPUT_FILE" | parallel -j $CONCURRENT_REQUESTS send_request
total_end_time=$(date +%s%N)

# Calculate and display total elapsed time
total_elapsed_time=$((total_end_time - total_start_time))
echo "========================================="
echo "Total Elapsed Time: $((total_elapsed_time / 1000000000)).$((total_elapsed_time % 1000000000 / 1000000)) seconds"
echo "========================================="
