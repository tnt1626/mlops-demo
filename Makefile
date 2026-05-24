install:
	uv sync

train:
	dvc repro retrain

train-base:
	dvc repro train

pipeline:
	dvc repro retrain

serve:
	uv run uvicorn src.api:app --reload

monitor:
	dvc repro monitor

retrain:
	dvc repro retrain
	
docker-build:
	docker build --platform linux/amd64 -t mlops-demo .
	
docker-run:
	docker run -p 8000:8000 -v "$$(pwd)/logs:/app/logs" mlops-demo
# 	2 dòng code dưới để lưu vào csv local, nếu k sẽ lưu vào csv của Docker, khi reload sẽ mất

simulate-drift:
	dvc repro simulate_drift
	
update-logs:
	gcloud storage cp gs://mlops-demo-hcmus-bucket/inference_logs.csv logs/inference_logs.csv
