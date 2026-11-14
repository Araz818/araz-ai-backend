# ARAZ AI Backend

FastAPI backend for ARAZ AI Android application.

## Deployment

Platform: Render Web Service (Python)

Root Directory: leave blank (repository root)

Build Command: `pip install -r backend/requirements.txt`

Start Command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`

## Environment Variables

GEMINI_API_KEY

ARAZ_ACCESS_TOKEN

GEMINI_MODEL (optional)

## API

POST /v1/chat

GET /health

## Security

Never commit GEMINI_API_KEY or ARAZ_ACCESS_TOKEN to GitHub.
Keep both secrets in Render environment variables.
