# ARAZ AI Backend

FastAPI backend for ARAZ AI Android application.

## Deployment

- Platform: Render
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT

## Environment Variables

- GEMINI_API_KEY
- ARAZ_ACCESS_TOKEN
- GEMINI_MODEL (optional)

## API

POST /v1/chat

GET /health

Never commit API keys or access tokens to GitHub.
