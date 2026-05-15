"""MCP server providing OCR and image generation via OpenAI APIs.

Designed for non-multimodal AI models that need assistance understanding
image content. Supports file paths, URLs, and text-to-image generation.

Supports two API modes:
  - chat (default):  Chat Completions API  (client.chat.completions)
  - responses:       Responses API         (client.responses)

Environment variables:
    OPENAI_API_KEY   (required)  OpenAI API key or compatible provider key.
    OPENAI_BASE_URL  (optional)  API base URL. Defaults to https://api.openai.com/v1.
    OPENAI_MODEL     (optional)  Model name. Defaults to gpt-4o.
    OPENAI_API_MODE  (optional)  API mode: "chat" or "responses".
                                 Defaults to "chat".
    OPENAI_IMAGE_MODEL       (optional)  Image generation model. Defaults to gpt-image-2.
    OPENAI_IMAGE_OUTPUT_DIR  (optional)  Directory for generated images. Defaults to generated_images.
"""

import base64
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

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
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-2")
IMAGE_OUTPUT_DIR = os.environ.get("OPENAI_IMAGE_OUTPUT_DIR", "generated_images")

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


def _image_extension(output_format: str) -> str:
    normalized = output_format.lower()
    return {
        "jpeg": "jpg",
        "jpg": "jpg",
        "png": "png",
        "webp": "webp",
    }.get(normalized, "png")


def _resolve_output_paths(output_path: str | None, count: int, output_format: str) -> list[Path]:
    ext = _image_extension(output_format)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if output_path:
        path = Path(output_path)
        if path.suffix:
            if count == 1:
                return [path]
            return [
                path.with_name(f"{path.stem}-{index}{path.suffix}")
                for index in range(1, count + 1)
            ]

        return [
            path / f"image-{timestamp}-{index}.{ext}"
            for index in range(1, count + 1)
        ]

    output_dir = Path(IMAGE_OUTPUT_DIR)
    run_id = uuid4().hex[:8]
    return [
        output_dir / f"image-{timestamp}-{run_id}-{index}.{ext}"
        for index in range(1, count + 1)
    ]


def _decode_image_item(item: dict) -> bytes:
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])

    if item.get("url"):
        resp = httpx.get(item["url"], timeout=90, follow_redirects=True)
        resp.raise_for_status()
        return resp.content

    raise RuntimeError("Image generation response did not include b64_json or url.")


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


def _call_image_generation(payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL.rstrip('/')}/images/generations"

    with httpx.Client(timeout=180) as http_client:
        response = http_client.post(url, headers=headers, json=payload)

    if response.status_code >= 400:
        body = response.text
        raise RuntimeError(f"API error {response.status_code}: {body}")

    return response.json()


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


@mcp.tool(
    description="Generate image files from a text prompt using the OpenAI image generation API."
)
def generate_image(
    prompt: str,
    output_path: str | None = None,
    size: str = "1024x1024",
    quality: str = "auto",
    output_format: str = "png",
    n: int = 1,
    background: str | None = None,
    user: str | None = None,
) -> str:
    """Generate images from a text prompt and save them locally.

    Args:
        prompt: Text prompt describing the image to generate.
        output_path: Optional output file path or directory. If omitted, images
                     are saved under OPENAI_IMAGE_OUTPUT_DIR.
        size: Image size, such as "1024x1024", "1024x1536", or "1536x1024".
        quality: Image quality, typically "auto", "low", "medium", or "high".
        output_format: Output format, typically "png", "jpeg", or "webp".
        n: Number of images to generate.
        background: Optional background mode if supported by the image model.
        user: Optional end-user identifier for API abuse monitoring.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    payload = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "n": n,
    }
    if background:
        payload["background"] = background
    if user:
        payload["user"] = user

    result = _call_image_generation(payload)
    items = result.get("data") or []
    if not items:
        raise RuntimeError("Image generation response did not include any images.")

    paths = _resolve_output_paths(output_path, len(items), output_format)
    saved_images = []
    for index, item in enumerate(items):
        image_bytes = _decode_image_item(item)
        path = paths[index]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_bytes)
        saved_images.append(
            {
                "path": str(path.resolve()),
                "revised_prompt": item.get("revised_prompt"),
            }
        )

    return json.dumps(
        {
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "images": saved_images,
            "usage": result.get("usage"),
        },
        ensure_ascii=False,
        indent=2,
    )


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
