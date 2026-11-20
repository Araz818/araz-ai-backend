# ARAZ AI Backend

FastAPI backend for ARAZ AI Android application.

## Deployment

- Platform: Render Web Service (Python)
- Root Directory: leave blank (repository root)
- Build Command: `pip install -r backend/requirements.txt`
- Start Command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`

## Environment Variables

- `ARAZ_ACCESS_TOKEN` authenticates requests from the Android app.
- `ARAZ_AI_PROVIDER` selects `gemini` (default) or `openai`.
- Gemini uses `GEMINI_API_KEY` and optionally `GEMINI_MODEL`.
- OpenAI uses `OPENAI_API_KEY` and optionally `OPENAI_MODEL` (default: `gpt-4.1-mini`).

To switch to GPT, set `ARAZ_AI_PROVIDER=openai` and add `OPENAI_API_KEY` in the Render service's Environment settings, then deploy. The app can keep using the same URL and access token.

Once deployed, check `GET /health` for `configured: true`, then send these messages in separate chats:

- Azerbaijani: “Sabah üçün qısa alış-veriş siyahısı yaz.” Expect an Azerbaijani shopping list.
- Turkish: “Yarın için kısa bir alışveriş listesi yaz.” Expect a Turkish shopping list.

## API

POST /v1/chat

GET /health

Never commit API keys or access tokens to GitHub.
