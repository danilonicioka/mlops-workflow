#!/bin/bash

# Variables
SERVICE_HOSTNAME="youtubegoes5g.kubeflow-user-example-com.svc.cluster.local"
INGRESS_HOST="localhost"
INGRESS_PORT="8085"
MODEL_NAME="youtubegoes5g"
INPUT_FILE="./input.json"

# Get the number of requests from the command line, default to 10 if not provided
REQUEST_COUNT=${1:-10}

# Initialize total response time (in milliseconds)
total_response_time_ms=0

echo "Starting sequential requests..."
echo "Number of requests to send: $REQUEST_COUNT"

for ((i=1; i<=REQUEST_COUNT; i++)); do
    echo "Sending request $i..."
    # Send request and capture response time in seconds
    response_time=$(curl -s -o /dev/null -w "%{time_total}" -H "Host: ${SERVICE_HOSTNAME}" \
                    -H "Content-Type: application/json" \
                    -d @"${INPUT_FILE}" \
                    http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict)
    # Convert response time to milliseconds (seconds * 1000)
    response_time_ms=$(echo "$response_time * 1000" | bc | awk '{printf "%.0f", $0}')
    # Add response time to total
    total_response_time_ms=$((total_response_time_ms + response_time_ms))

    echo "Response $i time: ${response_time_ms} ms"
done

# Display total response time in milliseconds
echo "-----------------------------------------"
echo "Total response time for $REQUEST_COUNT requests: ${total_response_time_ms} ms"
