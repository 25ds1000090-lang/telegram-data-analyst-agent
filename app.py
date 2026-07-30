from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from gemini_agent import solve_question


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "",
).strip()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "",
).strip().rstrip("/")

WEBHOOK_SECRET = os.getenv(
    "WEBHOOK_SECRET",
    "",
).strip()

WEBHOOK_PATH = "/telegram/webhook"

TELEGRAM_API_BASE = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
)

MAX_HISTORY_MESSAGES = int(
    os.getenv("MAX_HISTORY_MESSAGES", "12")
)

MAX_TELEGRAM_MESSAGE_LENGTH = 4000

LOG_DIRECTORY = Path(
    os.getenv("LOG_DIRECTORY", "logs")
)

LOG_RETENTION_SECONDS = int(
    os.getenv("LOG_RETENTION_SECONDS", "86400")
)

DROP_PENDING_UPDATES = (
    os.getenv("DROP_PENDING_UPDATES", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "%(name)s %(message)s"
    ),
)

logger = logging.getLogger(
    "telegram-data-analyst-agent"
)

# Prevent httpx from printing Telegram URLs containing the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Telegram Data Analyst Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

LOG_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

# Stores recent user messages separately for each Telegram chat.
conversation_history: dict[int, deque[str]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
)

# Prevents simultaneous processing of two messages in the same chat.
chat_locks: dict[int, asyncio.Lock] = defaultdict(
    asyncio.Lock
)

# Prevents Telegram duplicate delivery from causing repeated Gemini calls.
processed_update_ids: set[int] = set()
processed_update_order: deque[int] = deque(
    maxlen=2000
)

# Keeps references to background tasks so Python does not discard them.
background_tasks: set[asyncio.Task[Any]] = set()


# ---------------------------------------------------------------------------
# JSONL run logger
# ---------------------------------------------------------------------------

class JsonlRunLogger:
    """
    Writes structured events to a public JSONL log file.

    The Render free filesystem is temporary. Logs may disappear after
    redeployment, restart, or instance replacement.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.path = LOG_DIRECTORY / f"{run_id}.jsonl"
        self._lock = asyncio.Lock()

    def write(
        self,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Synchronous write method expected by gemini_agent.py.
        """
        record = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "event": event,
            "data": redact_sensitive_data(data or {}),
        }

        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(encoded + "\n")


# ---------------------------------------------------------------------------
# Security and redaction helpers
# ---------------------------------------------------------------------------

def redact_string(value: str) -> str:
    """
    Redact known secrets if they accidentally appear in log content.
    """
    redacted = value

    secrets = [
        TELEGRAM_BOT_TOKEN,
        GEMINI_API_KEY,
        WEBHOOK_SECRET,
    ]

    for secret in secrets:
        if secret:
            redacted = redacted.replace(
                secret,
                "[REDACTED]",
            )

    # Defensive Telegram-token pattern redaction.
    redacted = re.sub(
        r"\b\d{7,12}:[A-Za-z0-9_-]{20,}\b",
        "[REDACTED_TELEGRAM_TOKEN]",
        redacted,
    )

    return redacted


def redact_sensitive_data(value: Any) -> Any:
    """
    Recursively redact secrets from dictionaries, arrays and strings.
    """
    if isinstance(value, str):
        return redact_string(value)

    if isinstance(value, list):
        return [
            redact_sensitive_data(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            redact_sensitive_data(item)
            for item in value
        ]

    if isinstance(value, dict):
        result: dict[str, Any] = {}

        for key, item in value.items():
            lowered_key = str(key).lower()

            if any(
                sensitive_word in lowered_key
                for sensitive_word in (
                    "token",
                    "secret",
                    "api_key",
                    "apikey",
                    "authorization",
                )
            ):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = redact_sensitive_data(
                    item
                )

        return result

    return value


def verify_webhook_secret(
    supplied_secret: str | None,
) -> bool:
    """
    Compare the Telegram secret header safely.

    When WEBHOOK_SECRET is empty, secret validation is disabled.
    """
    if not WEBHOOK_SECRET:
        return True

    if not supplied_secret:
        return False

    return hmac.compare_digest(
        supplied_secret,
        WEBHOOK_SECRET,
    )


def validate_environment() -> None:
    """
    Log missing configuration and reject invalid webhook secrets.
    """
    missing: list[str] = []

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if not PUBLIC_BASE_URL:
        missing.append("PUBLIC_BASE_URL")

    if missing:
        logger.warning(
            "Missing environment variables: %s",
            ", ".join(missing),
        )

    if WEBHOOK_SECRET:
        if not re.fullmatch(
            r"[A-Za-z0-9_-]{1,256}",
            WEBHOOK_SECRET,
        ):
            raise RuntimeError(
                "WEBHOOK_SECRET contains invalid characters. "
                "Use only A-Z, a-z, 0-9, underscore and hyphen."
            )


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

async def telegram_api_request(
    method: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Call a Telegram Bot API method without logging the secret URL.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured."
        )

    url = f"{TELEGRAM_API_BASE}/{method}"

    timeout = httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=30.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:
        response = await client.post(
            url,
            json=payload,
        )

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Telegram returned non-JSON HTTP "
            f"{response.status_code}."
        ) from exc

    if response.status_code >= 400:
        description = response_data.get(
            "description",
            "Unknown Telegram error",
        )

        raise RuntimeError(
            f"Telegram API HTTP "
            f"{response.status_code}: {description}"
        )

    if not response_data.get("ok"):
        raise RuntimeError(
            "Telegram API rejected the request: "
            f"{response_data.get('description', 'unknown error')}"
        )

    return response_data


def split_telegram_message(
    text: str,
) -> list[str]:
    """
    Split long responses into Telegram-safe message sizes.
    """
    if len(text) <= MAX_TELEGRAM_MESSAGE_LENGTH:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= MAX_TELEGRAM_MESSAGE_LENGTH:
            chunks.append(remaining)
            break

        split_at = remaining.rfind(
            "\n",
            0,
            MAX_TELEGRAM_MESSAGE_LENGTH,
        )

        if split_at < 1000:
            split_at = remaining.rfind(
                " ",
                0,
                MAX_TELEGRAM_MESSAGE_LENGTH,
            )

        if split_at < 1000:
            split_at = MAX_TELEGRAM_MESSAGE_LENGTH

        chunks.append(
            remaining[:split_at].rstrip()
        )

        remaining = remaining[split_at:].lstrip()

    return chunks


async def send_telegram_message(
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
) -> None:
    """
    Send plain-text messages to a Telegram chat.
    """
    for index, chunk in enumerate(
        split_telegram_message(text)
    ):
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_web_page_preview": True,
        }

        if index == 0 and reply_to_message_id:
            payload["reply_parameters"] = {
                "message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
            }

        await telegram_api_request(
            "sendMessage",
            payload,
        )


async def register_telegram_webhook() -> None:
    """
    Register the Render webhook with Telegram at application startup.
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.warning(
            "Webhook registration skipped: "
            "TELEGRAM_BOT_TOKEN is missing."
        )
        return

    if not PUBLIC_BASE_URL:
        logger.warning(
            "Webhook registration skipped: "
            "PUBLIC_BASE_URL is missing."
        )
        return

    webhook_url = (
        f"{PUBLIC_BASE_URL}{WEBHOOK_PATH}"
    )

    payload: dict[str, Any] = {
        "url": webhook_url,
        "allowed_updates": [
            "message",
            "edited_message",
        ],
        "drop_pending_updates": DROP_PENDING_UPDATES,
    }

    if WEBHOOK_SECRET:
        payload["secret_token"] = WEBHOOK_SECRET

    try:
        result = await telegram_api_request(
            "setWebhook",
            payload,
        )

        logger.info(
            "Telegram webhook registered: %s",
            bool(result.get("ok")),
        )

    except Exception:
        logger.exception(
            "Telegram webhook registration failed."
        )


# ---------------------------------------------------------------------------
# Update deduplication
# ---------------------------------------------------------------------------

def mark_update_as_processed(
    update_id: int,
) -> bool:
    """
    Return False if the update was already received.

    Return True and store it when it is new.
    """
    if update_id in processed_update_ids:
        return False

    if (
        processed_update_order.maxlen
        and len(processed_update_order)
        >= processed_update_order.maxlen
    ):
        oldest_id = processed_update_order.popleft()
        processed_update_ids.discard(oldest_id)

    processed_update_order.append(update_id)
    processed_update_ids.add(update_id)

    return True


def keep_background_task(
    task: asyncio.Task[Any],
) -> None:
    """
    Retain a background task and remove it after completion.
    """
    background_tasks.add(task)

    task.add_done_callback(
        background_tasks.discard
    )


# ---------------------------------------------------------------------------
# Update processing
# ---------------------------------------------------------------------------

def extract_message(
    update: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Extract a normal or edited Telegram message.
    """
    message = update.get("message")

    if isinstance(message, dict):
        return message

    edited_message = update.get(
        "edited_message"
    )

    if isinstance(edited_message, dict):
        return edited_message

    return None


def create_public_log_url(
    run_id: str,
) -> str:
    if not PUBLIC_BASE_URL:
        return f"/logs/{run_id}.jsonl"

    return (
        f"{PUBLIC_BASE_URL}/logs/"
        f"{run_id}.jsonl"
    )


def serialize_answer_for_telegram(
    answer: Any,
    log_url: str,
) -> str:
    """
    Create the assignment's final JSON response.
    """
    response_object = {
        "answer": answer,
        "log_url": log_url,
    }

    return json.dumps(
        response_object,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        default=str,
    )


async def process_telegram_update(
    update: dict[str, Any],
) -> None:
    """
    Process one Telegram update in the background.

    Any exception is caught here so Telegram does not repeatedly redeliver
    the update.
    """
    message = extract_message(update)

    if not message:
        return

    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    message_id = message.get("message_id")
    text = message.get("text")

    if not isinstance(chat_id, int):
        return

    if not isinstance(text, str) or not text.strip():
        await send_telegram_message(
            chat_id=chat_id,
            text=(
                "Please send a text question or provide "
                "the data in your message."
            ),
            reply_to_message_id=message_id,
        )
        return

    clean_text = text.strip()

    # Avoid treating Telegram commands as data-analysis prompts.
    if clean_text.lower() in {
        "/start",
        "/help",
    }:
        await send_telegram_message(
            chat_id=chat_id,
            text=(
                "Send me a data-analysis question, calculation, "
                "table, or dataset description. I will return "
                "the result as JSON."
            ),
            reply_to_message_id=message_id,
        )
        return

    run_id = uuid.uuid4().hex
    run_logger = JsonlRunLogger(run_id)
    log_url = create_public_log_url(run_id)

    run_logger.write(
        "telegram_update_received",
        {
            "update_id": update.get("update_id"),
            "chat_id_hash": hashlib.sha256(
                str(chat_id).encode("utf-8")
            ).hexdigest()[:16],
            "message_id": message_id,
            "text_length": len(clean_text),
        },
    )

    # One message at a time per chat prevents race conditions in history.
    async with chat_locks[chat_id]:
        history = conversation_history[chat_id]
        history.append(clean_text)

        messages = list(history)

        try:
            answer = await solve_question(
                messages=messages,
                run_logger=run_logger,
            )

            result_text = serialize_answer_for_telegram(
                answer=answer,
                log_url=log_url,
            )

            await send_telegram_message(
                chat_id=chat_id,
                text=result_text,
                reply_to_message_id=message_id,
            )

            run_logger.write(
                "telegram_response_sent",
                {
                    "success": True,
                },
            )

        except Exception as exc:
            logger.exception(
                "Telegram update processing failed."
            )

            run_logger.write(
                "processing_failed",
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )

            error_answer = {
                "error": (
                    "The analysis could not be completed."
                )
            }

            error_text = serialize_answer_for_telegram(
                answer=error_answer,
                log_url=log_url,
            )

            try:
                await send_telegram_message(
                    chat_id=chat_id,
                    text=error_text,
                    reply_to_message_id=message_id,
                )
            except Exception:
                logger.exception(
                    "Failed to send Telegram error response."
                )


# ---------------------------------------------------------------------------
# Log cleanup
# ---------------------------------------------------------------------------

def cleanup_old_logs() -> None:
    """
    Remove expired local JSONL logs.
    """
    cutoff = (
        time.time() - LOG_RETENTION_SECONDS
    )

    for path in LOG_DIRECTORY.glob("*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove old log: %s",
                path.name,
            )


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def application_startup() -> None:
    validate_environment()
    cleanup_old_logs()
    await register_telegram_webhook()


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.api_route(
    "/",
    methods=["GET", "HEAD"],
)
async def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "telegram-data-analyst-agent",
        "telegram_token_configured": bool(
            TELEGRAM_BOT_TOKEN
        ),
        "gemini_key_configured": bool(
            GEMINI_API_KEY
        ),
        "public_base_url_configured": bool(
            PUBLIC_BASE_URL
        ),
        "webhook_secret_enabled": bool(
            WEBHOOK_SECRET
        ),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "telegram_token_configured": bool(
            TELEGRAM_BOT_TOKEN
        ),
        "gemini_key_configured": bool(
            GEMINI_API_KEY
        ),
        "public_base_url_configured": bool(
            PUBLIC_BASE_URL
        ),
    }


@app.post(WEBHOOK_PATH)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(
        default=None
    ),
) -> JSONResponse:
    """
    Acknowledge Telegram immediately.

    Gemini processing runs as a background asyncio task, so Telegram does
    not resend the same update while waiting for the model.
    """
    if not verify_webhook_secret(
        x_telegram_bot_api_secret_token
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook secret.",
        )

    try:
        update = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body.",
        ) from exc

    if not isinstance(update, dict):
        raise HTTPException(
            status_code=400,
            detail="Telegram update must be a JSON object.",
        )

    update_id = update.get("update_id")

    if not isinstance(update_id, int):
        # Still acknowledge malformed or irrelevant Telegram updates.
        return JSONResponse(
            status_code=200,
            content={"ok": True},
        )

    if not mark_update_as_processed(update_id):
        logger.info(
            "Ignored duplicate Telegram update: %s",
            update_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "duplicate": True,
            },
        )

    task = asyncio.create_task(
        process_telegram_update(update)
    )

    keep_background_task(task)

    # Telegram receives HTTP 200 immediately.
    return JSONResponse(
        status_code=200,
        content={"ok": True},
    )


@app.get("/logs/{filename}")
async def get_log(
    filename: str,
) -> FileResponse:
    """
    Serve only UUID-style JSONL log filenames.
    """
    if not re.fullmatch(
        r"[a-f0-9]{32}\.jsonl",
        filename,
    ):
        raise HTTPException(
            status_code=404,
            detail="Log not found.",
        )

    log_path = LOG_DIRECTORY / filename

    if not log_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Log not found.",
        )

    return FileResponse(
        path=log_path,
        media_type="application/x-ndjson",
        filename=filename,
    )
