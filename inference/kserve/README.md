# Inference Request

export INGRESS_HOST=localhost
export INGRESS_PORT=8085
MODEL_NAME=youtubegoes5g
SERVICE_HOSTNAME=$(kubectl get inferenceservice youtubegoes5g -o jsonpath='{.status.url}' | cut -d "/" -f 3)

# Types of tests

## Single request
```
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./input.json -w "\nResponse Time: %{time_total}s\n" 
```

## Batched - Single request, but multiple data
```
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./inputs.json -w "\nResponse Time: %{time_total}s\n"
```

or

```
bash batched.sh
```

## Sequential requests - Multiple requests one after another

```
bash sequential.sh
```

## Concurrent request

```
bash concurrent.sh
```

## Stress Testing

ab -n 1000 -c 10 -p batched.json -T "application/json" http://localhost:8085/v1/models/youtubegoes5g:predict