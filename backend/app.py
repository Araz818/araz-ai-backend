
import hmac
import os
import time
from collections import defaultdict, deque
from typing import Literal

import httpx2 as httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ARAZ_ACCESS_TOKEN = os.getenv("ARAZ_ACCESS_TOKEN", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

app = FastAPI(title="ARAZ AI Gemini API", version="1.2.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ARAZ_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

_rate_windows = defaultdict(deque)
MAX_REQUESTS_PER_MINUTE = 20
MAX_HISTORY_MESSAGES = 24

SYSTEM_INSTRUCTION = (
    "Sən ARAZ AI adlı şəxsi köməkçisən. Sakit, peşəkar, mehriban və məntiqli danış. "
    "Son istifadəçi mesajının dilini müəyyən et və Azərbaycan dilindədirsə Azərbaycan dilində, "
    "Türk dilindədirsə Türkiyə türkcəsində cavab ver. Dil aydın deyilsə söhbətin dilini izləyərək "
    "Azərbaycan dilini seç. Əvvəlki mesajların kontekstini nəzərə al. Bilmədiyin məlumatı uydurma; "
    "internetə çıxışın olmadığını və ya qeyri-müəyyənliyi açıq bildir. Təhlükəli və ya maliyyə "
    "əməliyyatlarını istifadəçinin açıq təsdiqi olmadan icra etdiyini iddia etmə."
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        min_length=1,
        max_length=MAX_HISTORY_MESSAGES
    )


class ChatResponse(BaseModel):
    reply: str


def authorize(authorization: str | None) -> None:
    if not GEMINI_API_KEY or not ARAZ_ACCESS_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Backend GEMINI_API_KEY və ARAZ_ACCESS_TOKEN ilə konfiqurasiya edilməyib.",
        )

    supplied_token = authorization.removeprefix("Bearer ") if authorization else ""

    if not hmac.compare_digest(supplied_token, ARAZ_ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="ARAZ giriş tokeni yanlışdır.")


def enforce_rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _rate_windows[client_host]

    while window and now - window[0] > 60:
        window.popleft()

    if len(window) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Sorğu limiti keçildi. Bir dəqiqə sonra yenidən cəhd edin."
        )

    window.append(now)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {
        "ok": True,
        "configured": bool(GEMINI_API_KEY and ARAZ_ACCESS_TOKEN)
    }


@app.post("/chat", response_model=ChatResponse)
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> ChatResponse:

    authorize(authorization)
    enforce_rate_limit(request)

    contents = [
        {
            "role": "model" if item.role == "assistant" else "user",
            "parts": [{"text": item.content}],
        }
        for item in body.messages[-MAX_HISTORY_MESSAGES:]
    ]

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.65,
            "maxOutputTokens": 1200
        },
    }

    try:
        async with httpx.AsyncClient(timeout=40) as client:
            response = await client.post(
                GEMINI_URL,
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json=payload,
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Gemini cavab verməyə gecikir. Yenidən cəhd edin."
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini xidmətinə qoşulmaq mümkün olmadı."
        ) from exc

    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="Gemini pulsuz kvota limiti bitib. Bir qədər sonra yenidən cəhd edin."
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail="Gemini sorğunu qəbul etmədi. Serverdəki model və Gemini açarını yoxlayın."
        )

    try:
        result = response.json()
        candidates = result.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []

        reply = "\n".join(
            part.get("text", "")
            for part in parts
            if isinstance(part.get("text"), str)
        ).strip()

    except (ValueError, IndexError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Gemini cavabının formatı tanınmadı."
        ) from exc

    if not reply:
        raise HTTPException(
            status_code=502,
            detail="Gemini cavab yarada bilmədi. Sorğunu fərqli yazın."
        )

    return ChatResponse(reply=reply)
