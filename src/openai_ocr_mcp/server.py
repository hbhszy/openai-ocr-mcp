"""MCP server providing OCR capability via OpenAI vision API.

Designed for non-multimodal AI models that need assistance understanding
image content. Supports file paths, URLs.

Environment variables:
    OPENAI_API_KEY  (required)  OpenAI API key or compatible provider key.
    OPENAI_BASE_URL (optional)  API base URL. Defaults to https://api.openai.com/v1.
    OPENAI_MODEL    (optional)  Model name. Defaults to gpt-4o.
"""

import base64
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


# ── MCP tool ─────────────────────────────────────────────────────────────

@mcp.tool(
    description="Analyse an image using the OpenAI vision API and return the text or description it contains."
)
def ocr_image(
    source: str,
    prompt: str = "Please read and describe all the text and visual content in this image in detail.",
    detail: str = "auto",
) -> str:
    """Analyse an image using the OpenAI vision API.

    Args:
        source: Local file path or HTTP(S) URL of the image.
        prompt: Custom instruction for the vision model.
        detail: Image detail level ("auto", "low", or "high").
    """
    media_type, b64_data = _load_image(source)
    data_url = f"data:{media_type};base64,{b64_data}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
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
        max_tokens=4096,
    )

    return response.choices[0].message.content or ""


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
