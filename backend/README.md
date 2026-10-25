# Backend (FastAPI)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # Windows: copy .env.example .env
python -m app.ml.train                               # trains + saves artifacts to models/
uvicorn app.main:app --reload --port 8000            # Swagger: http://localhost:8000/docs
pytest -q
```

Environment variables (see `.env.example`): `CORS_ORIGINS`, `DATA_PATH`, `MODEL_DIR`.
Assessment history is kept in memory and resets when the server restarts.
