labels:
	python -m src.labels
frames:
	python scripts/extract_frames.py --fps 10
frames-around:
	python scripts/extract_frames.py --fps 10 --around 3
eval:
	python scripts/evaluate.py
serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
tunnel:
	cloudflared tunnel --url http://localhost:8000
docker:
	docker compose up --build
check:
	python scripts/check_env.py
