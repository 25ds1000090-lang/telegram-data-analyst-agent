from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from gemini_agent import solve_question
from logger import RunLogger, get_log_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Example: https://telegram-data-analyst-agent.onrender.com
PUBLIC_BASE_URL = (
    os.getenv("PUBLIC_BASE_URL")
    or os.getenv("WEBHOOK_URL")
    or ""
).strip().rstrip("/")

# Telegram sends this secret in the X-Telegram-Bot-Api-Secret-Token header.
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
TELEGRAM_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "20"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("telegram-data-analyst-agent")


# ---------------------------------------------------------------------------
# In-memory conversation state
# ---------------------------------------------------------------------------

# Each Telegram chat gets an independent short history.
conversation_history: dict[int, deque[str]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
)

# Avoid processing two messages from the same chat simultaneously.
chat_locks: dict[int, asyncio.Lock] = {}


def get_chat_lock(chat_id: int) -> asyncio.Lock:
    lock = chat_locks.get(chat_id)

    if lock is None:
        lock = asyncio.Lock()
        chat_locks[chat_id] = lock

    return lock


# Telegram update IDs are used to prevent accidental duplicate processing.
processed_update_ids: deque[int] = deque(maxlen=2_000)
processed_update_id_set: set[int] = set()
processed_updates_lock = asyncio.Lock()


async def claim_update(update_id: int) -> bool:
    """
    Return True when this update has not been handled before.

    Telegram can retry webhook deliveries. This prevents duplicate replies.
    """
    async with processed_updates_lock:
        if update_id in processed_update_id_set:
            return False

        if len(processed_update_ids) == processed_update_ids.maxlen:
            oldest = processed_update_ids.popleft()
            processed_update_id_set.discard(oldest)

        processed_update_ids.append(update_id)
        processed_update_id_set.add(update_id)
        return True


# ---------------------------------------------------------------------------
# Telegram API helpers
# ---------------------------------------------------------------------------

def telegram_api_url(method: str) -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")

    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


async def telegram_request(
    method: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Call the Telegram Bot API and return its JSON response.
    """
    timeout = httpx.Timeout(TELEGRAM_TIMEOUT_SECONDS)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            telegram_api_url(method),
            json=payload or {},
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Telegram returned non-JSON response with HTTP "
            f"{response.status_code}."
        ) from exc

    if response.status_code >= 400 or not body.get("ok"):
        description = body.get("description", "Unknown Telegram API error")
        raise RuntimeError(
            f"Telegram {method} failed with HTTP "
            f"{response.status_code}: {description}"
        )

    return body


async def send_telegram_json(
    chat_id: int,
    reply_object: dict[str, Any],
) -> None:
    """
    Send exactly one compact JSON object and no surrounding explanation.
    """
    reply_text = json.dumps(
        reply_object,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    await telegram_request(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": reply_text,
            # No Markdown or HTML parse mode: Telegram receives plain JSON.
            "disable_web_page_preview": True,
        },
    )


async def register_webhook() -> None:
    """
    Register the deployed HTTPS webhook with Telegram.
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.warning(
            "TELEGRAM_BOT_TOKEN is missing; webhook was not registered."
        )
        return

    if not PUBLIC_BASE_URL:
        logger.warning(
            "PUBLIC_BASE_URL/WEBHOOK_URL is missing; webhook was not registered."
        )
        return

    webhook_url = f"{PUBLIC_BASE_URL}/telegram/webhook"

    payload: dict[str, Any] = {
        "url": webhook_url,
        "allowed_updates": ["message", "edited_message"],
        "drop_pending_updates": False,
    }

    if WEBHOOK_SECRET:
        payload["secret_token"] = WEBHOOK_SECRET

    result = await telegram_request("setWebhook", payload)
    logger.info("Telegram webhook registered: %s", result.get("result"))


# ---------------------------------------------------------------------------
# Telegram update extraction
# ---------------------------------------------------------------------------

class TelegramMessage(BaseModel):
    message_id: int
    chat: dict[str, Any]
    text: str | None = None
    date: int | None = None


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage | None = None
    edited_message: TelegramMessage | None = None


def extract_message(
    update: TelegramUpdate,
) -> tuple[int, int, str] | None:
    """
    Extract chat ID, message ID and text from a Telegram update.
    """
    message = update.message or update.edited_message

    if message is None:
        return None

    chat_id = message.chat.get("id")

    if not isinstance(chat_id, int):
        return None

    text = (message.text or "").strip()

    if not text:
        return None

    return chat_id, message.message_id, text


# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------

async def process_message(
    *,
    update_id: int,
    chat_id: int,
    message_id: int,
    text: str,
) -> None:
    """
    Analyse one Telegram message and return one JSON object.
    """
    async with get_chat_lock(chat_id):
        history = conversation_history[chat_id]
        history.append(text)

        run_logger = RunLogger(
            chat_id=chat_id,
            update_id=update_id,
            message_id=message_id,
        )

        run_logger.write(
            "request",
            {
                "message": text,
                "history_size": len(history),
            },
        )

        try:
            answer = await solve_question(
                messages=list(history),
                run_logger=run_logger,
            )

            run_logger.write(
                "answer_ready",
                {
                    "answer": answer,
                },
            )

        except Exception as exc:
            logger.exception(
                "Analysis failed for chat_id=%s update_id=%s",
                chat_id,
                update_id,
            )

            run_logger.write(
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )

            # This is only a last-resort response. Successful graded questions
            # should return the schema requested by the question.
            answer = {
                "error": "The analysis could not be completed."
            }

        log_url = f"{PUBLIC_BASE_URL}/logs/{run_logger.run_id}.jsonl"

        final_reply = {
            "answer": answer,
            "log_url": log_url,
        }

        run_logger.write(
            "final_reply",
            {
                "reply": final_reply,
            },
        )

        await send_telegram_json(chat_id, final_reply)

        run_logger.write(
            "telegram_delivery",
            {
                "status": "sent",
            },
        )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    missing = []

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if not PUBLIC_BASE_URL:
        missing.append("PUBLIC_BASE_URL or WEBHOOK_URL")

    if missing:
        logger.warning(
            "Missing environment variables: %s",
            ", ".join(missing),
        )

    try:
        await register_webhook()
    except Exception:
        # Do not stop the web server from starting. The health endpoint and
        # Render logs remain available for troubleshooting.
        logger.exception("Automatic Telegram webhook registration failed.")

    yield


app = FastAPI(
    title="Telegram Data Analyst Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "telegram-data-analyst-agent",
        "status": "ok",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "telegram_token_configured": bool(TELEGRAM_BOT_TOKEN),
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "public_base_url_configured": bool(PUBLIC_BASE_URL),
    }


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(
        default=None,
        alias="X-Telegram-Bot-Api-Secret-Token",
    ),
) -> JSONResponse:
    """
    Receive Telegram webhook updates.

    The response is returned immediately so Telegram does not have to wait
    while Gemini performs the analysis.
    """
    if WEBHOOK_SECRET:
        supplied_secret = x_telegram_bot_api_secret_token or ""

        if not secrets.compare_digest(
            supplied_secret,
            WEBHOOK_SECRET,
        ):
            raise HTTPException(
                status_code=403,
                detail="Invalid Telegram webhook secret.",
            )

    try:
        payload = await request.json()
        update = TelegramUpdate.model_validate(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid Telegram update.",
        ) from exc

    is_new = await claim_update(update.update_id)

    if not is_new:
        return JSONResponse(
            {
                "ok": True,
                "duplicate": True,
            }
        )

    extracted = extract_message(update)

    if extracted is None:
        return JSONResponse(
            {
                "ok": True,
                "ignored": True,
            }
        )

    chat_id, message_id, text = extracted

    asyncio.create_task(
        process_message(
            update_id=update.update_id,
            chat_id=chat_id,
            message_id=message_id,
            text=text,
        )
    )

    return JSONResponse({"ok": True})


@app.get(
    "/logs/{run_id}.jsonl",
    response_class=PlainTextResponse,
)
async def public_run_log(run_id: str) -> PlainTextResponse:
    """
    Return a publicly wget-able JSONL log for a specific run.
    """
    if not run_id or len(run_id) > 100:
        raise HTTPException(status_code=404, detail="Log not found.")

    # Prevent path traversal or unexpected filenames.
    if not all(character.isalnum() or character in "-_" for character in run_id):
        raise HTTPException(status_code=404, detail="Log not found.")

    log_text = get_log_text(run_id)

    if log_text is None:
        raise HTTPException(status_code=404, detail="Log not found.")

    return PlainTextResponse(
        content=log_text,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/run.jsonl", response_class=PlainTextResponse)
async def latest_run_log() -> PlainTextResponse:
    """
    Compatibility endpoint exposing the most recently created run log.
    """
    log_text = get_log_text(None)

    if log_text is None:
        return PlainTextResponse(
            content="",
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    return PlainTextResponse(
        content=log_text,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/admin/set-webhook")
async def set_webhook_manually(
    x_admin_secret: str | None = Header(
        default=None,
        alias="X-Admin-Secret",
    ),
) -> dict[str, Any]:
    """
    Optional troubleshooting endpoint.

    To use it safely, set ADMIN_SECRET on Render and send that value in the
    X-Admin-Secret header.
    """
    admin_secret = os.getenv("ADMIN_SECRET", "").strip()

    if not admin_secret:
        raise HTTPException(
            status_code=404,
            detail="Endpoint not configured.",
        )

    if not secrets.compare_digest(
        x_admin_secret or "",
        admin_secret,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid admin secret.",
        )

    await register_webhook()

    return {
        "ok": True,
        "webhook": f"{PUBLIC_BASE_URL}/telegram/webhook",
    }
