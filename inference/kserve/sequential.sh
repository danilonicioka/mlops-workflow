#!/bin/bash

URL="http://localhost:8085/v1/models/youtubegoes5g:predict"
INPUT_FILE="sequential.txt"
HOST="youtubegoes5g.kubeflow-user-example-com.svc.cluster.local"

# Start the timer for total processing
total_start_time=$(date +%s%N)

while IFS= read -r payload || [ -n "$payload" ]; do
    echo "Sending request with payload: $payload"
    start_time=$(date +%s%N)
    response=$(curl -s -X POST -H "Host: $HOST" -H "Content-Type: application/json" -d "$payload" $URL)
    end_time=$(date +%s%N)
    elapsed_time=$((end_time - start_time))
    echo "Response: $response"
    echo "Response Time: $((elapsed_time / 1000000)) ms"
    echo "-----------------------------------------"
done < "$INPUT_FILE"

# End the timer for total processing
total_end_time=$(date +%s%N)
total_elapsed_time=$((total_end_time - total_start_time))

# Show the total time for all requests
echo "Total Time for All Requests: $((total_elapsed_time / 1000000)) ms"
