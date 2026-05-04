"""MCP server providing OCR capability via OpenAI vision API.

Designed for non-multimodal AI models that need assistance understanding
image content. Supports file paths, URLs.

Supports two API modes:
  - chat (default):  Chat Completions API  (client.chat.completions)
  - responses:       Responses API         (client.responses)

Environment variables:
    OPENAI_API_KEY   (required)  OpenAI API key or compatible provider key.
    OPENAI_BASE_URL  (optional)  API base URL. Defaults to https://api.openai.com/v1.
    OPENAI_MODEL     (optional)  Model name. Defaults to gpt-4o.
    OPENAI_API_MODE  (optional)  API mode: "chat" or "responses".
                                 Defaults to "chat".
"""

import base64
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

load_dotenv()

# ── configuration ──────────────────────────────────────────────────────────

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
API_MODE = os.environ.get("OPENAI_API_MODE", "chat")

if not API_KEY:
    print("FATAL: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
mcp = FastMCP("openai-ocr-mcp")


# ── helpers ────────────────────────────────────────────────────────────────

def _image_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")


def _load_image(source: str) -> tuple[str, str]:
    """Load image from a file path or URL.

    Returns a tuple of (media_type, base64_data).
    """
    if source.startswith(("http://", "https://")):
        resp = httpx.get(source, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        media_type = resp.headers.get("content-type", "image/png")
        raw = resp.content
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        media_type = _image_media_type(path)
        raw = path.read_bytes()

    return media_type, base64.b64encode(raw).decode("utf-8")


# ── API callers ───────────────────────────────────────────────────────────

def _stream_text(path: str, payload: dict, stream_type: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    parts: list[str] = []

    with httpx.stream("POST", url, headers=headers, json=payload, timeout=90) as response:
        if response.status_code >= 400:
            body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API error {response.status_code}: {body}")

        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue

            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            if stream_type == "chat":
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        parts.append(delta["content"])
            elif event.get("type") == "response.output_text.delta":
                parts.append(event.get("delta") or "")
            elif event.get("type") == "response.failed":
                error = (event.get("response") or {}).get("error") or {}
                message = error.get("message") or "unknown streaming response error"
                raise RuntimeError(f"API error: {message}")

    return "".join(parts)


def _call_chat(prompt: str, data_url: str, detail: str) -> str:
    return _stream_text(
        "chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": detail},
                        },
                    ],
                },
            ],
            "stream": True,
        },
        "chat",
    )


def _responses_payload(prompt: str, data_url: str, detail: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": data_url,
                    "detail": detail,
                },
            ],
        },
    ]


def _call_responses(prompt: str, data_url: str, detail: str) -> str:
    return _stream_text(
        "responses",
        {
            "model": MODEL,
            "input": _responses_payload(prompt, data_url, detail),
            "stream": True,
        },
        "responses",
    )


# ── MCP tool ─────────────────────────────────────────────────────────────

@mcp.tool(
    description="Analyse an image using the OpenAI vision API and return the text or description it contains."
)
def ocr_image(
    source: str,
    prompt: str = "Please read and describe all the text and visual content in this image in detail.",
    detail: str = "auto",
    api_mode: str | None = None,
) -> str:
    """Analyse an image using the OpenAI vision API.

    Args:
        source: Local file path or HTTP(S) URL of the image.
        prompt: Custom instruction for the vision model.
        detail: Image detail level ("auto", "low", or "high").
        api_mode: API mode override ("chat" or "responses"). Falls back to
                  OPENAI_API_MODE env var, then "chat".
    """
    mode = api_mode or API_MODE
    caller = {
        "chat": _call_chat,
        "responses": _call_responses,
    }.get(mode)
    if caller is None:
        raise ValueError(f"Unsupported API mode: {mode}")

    media_type, b64_data = _load_image(source)
    data_url = f"data:{media_type};base64,{b64_data}"
    return caller(prompt, data_url, detail)


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
