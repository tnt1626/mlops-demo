install:
	uv sync

train:
	dvc exp run

serve:
	uv run uvicorn src.api:app --reload

monitor:
	uv run python src/monitor.py
	
docker-build:
	docker build --platform linux/amd64 -t mlops-demo .
	
docker-run:
	docker run -p 8000:8000 -v "$$(pwd)/logs:/app/logs" mlops-demo
# 	2 dòng code dưới để lưu vào csv local, nếu k sẽ lưu vào csv của Docker, khi reload sẽ mất

drift-simulate:
	uv run python src/simulate_request.py
	