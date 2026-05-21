"""MCP server providing OCR and image generation via OpenAI APIs.

Designed for non-multimodal AI models that need assistance understanding
image content. Supports file paths, URLs, text-to-image generation, and
editing existing images.

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
from contextlib import ExitStack, contextmanager
from datetime import datetime
from io import BytesIO
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


def _extension_for_media_type(media_type: str) -> str:
    media_type = media_type.split(";")[0].strip().lower()
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
    }.get(media_type, "png")


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


def _image_data_url(source: str) -> str:
    media_type, b64_data = _load_image(source)
    return f"data:{media_type};base64,{b64_data}"


def _image_reference(source: str) -> dict:
    if source.startswith(("http://", "https://", "data:")):
        return {"image_url": source}
    return {"image_url": _image_data_url(source)}


def _normalize_image_sources(source: str | list[str]) -> list[str]:
    if isinstance(source, str):
        return [source]
    if not source:
        raise ValueError("source must include at least one image.")
    return source


def _is_local_image_source(source: str) -> bool:
    return not source.startswith(("http://", "https://", "data:"))


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


def _item_get(item: object, key: str, default: object = None) -> object:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _result_get(result: object, key: str, default: object = None) -> object:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _decode_image_item(item: dict) -> bytes:
    b64_json = _item_get(item, "b64_json")
    if b64_json:
        return base64.b64decode(str(b64_json))

    url = _item_get(item, "url")
    if url:
        resp = httpx.get(str(url), timeout=90, follow_redirects=True)
        resp.raise_for_status()
        return resp.content

    raise RuntimeError("Image generation response did not include b64_json or url.")


@contextmanager
def _open_images_for_edit(sources: list[str]):
    """Open local paths as files and remote/data sources as in-memory files."""
    with ExitStack() as stack:
        files = []
        for index, source in enumerate(sources, start=1):
            if _is_local_image_source(source):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {source}")
                files.append(stack.enter_context(path.open("rb")))
                continue

            media_type, b64_data = _load_image(source)
            buffer = BytesIO(base64.b64decode(b64_data))
            buffer.name = f"image-{index}.{_extension_for_media_type(media_type)}"
            files.append(buffer)

        yield files


@contextmanager
def _open_mask_for_edit(mask: str | None):
    if not mask:
        yield None
        return

    if _is_local_image_source(mask):
        path = Path(mask)
        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask}")
        with path.open("rb") as handle:
            yield handle
        return

    media_type, b64_data = _load_image(mask)
    buffer = BytesIO(base64.b64decode(b64_data))
    buffer.name = f"mask.{_extension_for_media_type(media_type)}"
    yield buffer


def _image_model_uses_implicit_high_fidelity() -> bool:
    return IMAGE_MODEL == "gpt-image-2"


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


def _call_image_edit(payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL.rstrip('/')}/images/edits"

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
            "usage": _json_safe(result.get("usage")),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(
    description="Edit existing image files using the OpenAI image editing API."
)
def edit_image(
    source: str | list[str],
    prompt: str,
    mask: str | None = None,
    output_path: str | None = None,
    size: str = "auto",
    quality: str = "auto",
    output_format: str = "png",
    n: int = 1,
    background: str | None = None,
    input_fidelity: str | None = None,
    moderation: str | None = None,
    output_compression: int | None = None,
    user: str | None = None,
) -> str:
    """Edit one or more input images from a prompt and save the results locally.

    Args:
        source: Local file path, HTTP(S) URL, data URL, or list of image
                sources to edit.
        prompt: Text prompt describing the desired edit.
        mask: Optional local path, HTTP(S) URL, or data URL for an edit mask.
        output_path: Optional output file path or directory. If omitted, images
                     are saved under OPENAI_IMAGE_OUTPUT_DIR.
        size: Output image size, such as "auto", "1024x1024",
              "1024x1536", or "1536x1024".
        quality: Output quality, typically "auto", "low", "medium", or "high".
        output_format: Output format, typically "png", "jpeg", or "webp".
        n: Number of edited images to generate.
        background: Optional background mode, such as "auto", "transparent",
                    or "opaque".
        input_fidelity: Optional fidelity level for the input image(s),
                        "high" or "low".
        moderation: Optional moderation level for GPT image models,
                    "auto" or "low".
        output_compression: Optional 0-100 compression level for jpeg/webp.
        user: Optional end-user identifier for API abuse monitoring.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if output_compression is not None and not 0 <= output_compression <= 100:
        raise ValueError("output_compression must be between 0 and 100.")

    request = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "n": n,
    }
    if background:
        request["background"] = background
    if input_fidelity and not _image_model_uses_implicit_high_fidelity():
        request["input_fidelity"] = input_fidelity
    if moderation:
        request["moderation"] = moderation
    if output_compression is not None:
        request["output_compression"] = output_compression
    if user:
        request["user"] = user

    sources = _normalize_image_sources(source)
    with _open_images_for_edit(sources) as image_files, _open_mask_for_edit(mask) as mask_file:
        request["image"] = image_files if len(image_files) > 1 else image_files[0]
        if mask_file is not None:
            request["mask"] = mask_file
        result = client.images.edit(**request)

    items = _result_get(result, "data") or []
    if not items:
        raise RuntimeError("Image edit response did not include any images.")

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
                "revised_prompt": _item_get(item, "revised_prompt"),
            }
        )

    return json.dumps(
        {
            "size": _result_get(result, "size", size),
            "quality": _result_get(result, "quality", quality),
            "output_format": _result_get(result, "output_format", output_format),
            "background": _result_get(result, "background"),
            "images": saved_images,
            "usage": _json_safe(_result_get(result, "usage")),
        },
        ensure_ascii=False,
        indent=2,
    )


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
