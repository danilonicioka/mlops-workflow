torch-model-archiver --model-name youtubegoes5g --version 1.0 --model-file model.py --serialized-file model_trained_artifact.pt --handler handler.py --requirements-file requirements.txt



kubectl cp youtubegoes5g.mar model-store-pod:/pv/model-store/ -c model-store -n kserve-test