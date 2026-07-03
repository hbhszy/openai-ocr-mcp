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
    OPENAI_MODEL     (optional)  Model name. Defaults to gpt-5.4.
    OPENAI_API_MODE  (optional)  API mode: "chat" or "responses".
                                 Defaults to "chat".
    OPENAI_OCR_MCP_CONFIG       (optional)  JSON config file path. Defaults to ./config.json if present.
    OPENAI_OCR_MCP_DISABLED_TOOLS (optional) Comma-separated tool names to hide from MCP clients.
    OPENAI_<TOOL>_API_KEY       (optional)  Tool-specific API key.
    OPENAI_<TOOL>_BASE_URL      (optional)  Tool-specific API base URL.
    OPENAI_<TOOL>_MODEL         (optional)  Tool-specific model.
    OPENAI_IMAGE_MODEL       (optional)  Image generation model. Defaults to gpt-image-2.
    OPENAI_IMAGE_OUTPUT_DIR  (optional)  Directory for generated images. Defaults to generated_images.
"""

import base64
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from pydantic import Field

load_dotenv()

# ── configuration ──────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OCR_MODEL = "gpt-5.4"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_REQUEST_TIMEOUT = 1200.0

BASE_URL = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", DEFAULT_OCR_MODEL)
API_MODE = os.environ.get("OPENAI_API_MODE", "chat")
IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
IMAGE_OUTPUT_DIR = os.environ.get("OPENAI_IMAGE_OUTPUT_DIR", "generated_images")
REQUEST_TIMEOUT = float(
    os.environ.get("OPENAI_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT))
)

mcp = FastMCP("openai-ocr-mcp")


@dataclass(frozen=True)
class ToolConfig:
    base_url: str
    api_key: str
    model: str
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT


@lru_cache(maxsize=1)
def _load_config_file() -> dict:
    config_path = os.environ.get("OPENAI_OCR_MCP_CONFIG")
    if not config_path:
        default_path = Path.cwd() / "config.json"
        if not default_path.exists():
            return {}
        path = default_path
    else:
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("OPENAI_OCR_MCP_CONFIG must point to a JSON object.")
    return data


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _config_value(section: dict, *names: str) -> str | None:
    for name in names:
        value = section.get(name)
        if value is not None and str(value) != "":
            return str(value)
    return None


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _secret_from_config(section: dict) -> str | None:
    api_key = _config_value(section, "api_key", "apiKey")
    if api_key:
        return api_key

    api_key_env = _config_value(section, "api_key_env", "apiKeyEnv")
    if api_key_env:
        return os.environ.get(api_key_env)

    return None


def _request_timeout_from_config(
    defaults: dict,
    tool_config: dict,
    env_prefixes: tuple[str, ...],
    legacy_envs: tuple[str, ...] = (),
) -> float:
    value = (
        _env_value(*(f"{prefix}_REQUEST_TIMEOUT" for prefix in env_prefixes))
        or _env_value(*legacy_envs)
        or os.environ.get("OPENAI_REQUEST_TIMEOUT")
        or os.environ.get("OPENAI_IMAGE_REQUEST_TIMEOUT")
        or _config_value(tool_config, "request_timeout", "requestTimeout", "timeout")
        or _config_value(
            defaults,
            "request_timeout",
            "requestTimeout",
            "timeout",
            "image_request_timeout",
            "imageRequestTimeout",
        )
    )
    if not value:
        return DEFAULT_REQUEST_TIMEOUT

    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid image request timeout: {value}") from exc

    if timeout <= 0:
        raise ValueError("Image request timeout must be greater than 0.")
    return timeout


def _tool_config_section(config: dict, *tool_names: str) -> dict:
    tools = _mapping(config.get("tools"))
    for tool_name in tool_names:
        section = tools.get(tool_name)
        if isinstance(section, dict):
            return section
    return {}


def _resolve_tool_config(
    *,
    tool_names: tuple[str, ...],
    env_prefixes: tuple[str, ...],
    default_model: str,
    legacy_model_envs: tuple[str, ...] = (),
    legacy_timeout_envs: tuple[str, ...] = (),
) -> ToolConfig:
    config = _load_config_file()
    defaults = _mapping(config.get("defaults"))
    tool_config = _tool_config_section(config, *tool_names)

    base_url = (
        _env_value(*(f"{prefix}_BASE_URL" for prefix in env_prefixes))
        or _config_value(tool_config, "base_url", "baseUrl")
        or os.environ.get("OPENAI_BASE_URL")
        or _config_value(defaults, "base_url", "baseUrl")
        or DEFAULT_BASE_URL
    )
    api_key = (
        _env_value(*(f"{prefix}_API_KEY" for prefix in env_prefixes))
        or _secret_from_config(tool_config)
        or os.environ.get("OPENAI_API_KEY")
        or _secret_from_config(defaults)
        or ""
    )
    model = (
        _env_value(*(f"{prefix}_MODEL" for prefix in env_prefixes))
        or _config_value(tool_config, "model")
        or _env_value(*legacy_model_envs)
        or os.environ.get("OPENAI_MODEL")
        or _config_value(defaults, "model")
        or default_model
    )
    request_timeout = _request_timeout_from_config(
        defaults,
        tool_config,
        env_prefixes,
        legacy_timeout_envs,
    )

    if not api_key:
        names = ", ".join(f"{prefix}_API_KEY" for prefix in env_prefixes)
        raise RuntimeError(
            f"API key is not configured. Set {names}, OPENAI_API_KEY, "
            "or an api_key/api_key_env value in OPENAI_OCR_MCP_CONFIG."
        )

    return ToolConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        request_timeout=request_timeout,
    )


def _disabled_tools() -> set[str]:
    """Return set of tool names that should be hidden from MCP clients."""
    disabled: set[str] = set()

    try:
        config = _load_config_file()
    except (FileNotFoundError, ValueError):
        config = {}

    for name in config.get("disabled_tools", []):
        name = str(name).strip()
        if name:
            disabled.add(name)

    tools = _mapping(config.get("tools"))
    for tool_name, section in tools.items():
        if isinstance(section, dict):
            if section.get("enabled") is True:
                disabled.discard(tool_name)
            elif section.get("enabled") is False:
                disabled.add(tool_name)

    env_disabled = os.environ.get("OPENAI_OCR_MCP_DISABLED_TOOLS", "")
    if env_disabled:
        for name in env_disabled.split(","):
            name = name.strip()
            if name:
                disabled.add(name)

    return disabled


def _resolve_ocr_config() -> ToolConfig:
    return _resolve_tool_config(
        tool_names=("ocr_image", "ocr"),
        env_prefixes=("OPENAI_OCR", "OPENAI_OCR_IMAGE"),
        default_model=DEFAULT_OCR_MODEL,
    )


def _resolve_generate_image_config() -> ToolConfig:
    return _resolve_tool_config(
        tool_names=("generate_image",),
        env_prefixes=("OPENAI_GENERATE_IMAGE",),
        default_model=DEFAULT_IMAGE_MODEL,
        legacy_model_envs=("OPENAI_IMAGE_MODEL",),
        legacy_timeout_envs=("OPENAI_IMAGE_REQUEST_TIMEOUT",),
    )


def _resolve_edit_image_config() -> ToolConfig:
    return _resolve_tool_config(
        tool_names=("edit_image",),
        env_prefixes=("OPENAI_EDIT_IMAGE",),
        default_model=DEFAULT_IMAGE_MODEL,
        legacy_model_envs=("OPENAI_IMAGE_MODEL",),
        legacy_timeout_envs=("OPENAI_IMAGE_REQUEST_TIMEOUT",),
    )


def _resolve_ocr_api_mode(api_mode: str | None) -> str:
    config = _load_config_file()
    defaults = _mapping(config.get("defaults"))
    tool_config = _tool_config_section(config, "ocr_image", "ocr")
    return (
        api_mode
        or _env_value("OPENAI_OCR_API_MODE", "OPENAI_OCR_IMAGE_API_MODE")
        or _config_value(tool_config, "api_mode", "apiMode")
        or os.environ.get("OPENAI_API_MODE")
        or _config_value(defaults, "api_mode", "apiMode")
        or "chat"
    )


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


def _load_image(source: str, timeout: float = DEFAULT_REQUEST_TIMEOUT, work_dir: str | None = None) -> tuple[str, str]:
    """Load image from a file path or URL.

    Args:
        source: Local file path or HTTP(S) URL.
        timeout: Request timeout for URL downloads.
        work_dir: Working directory for resolving relative local paths.
                  If None, relative paths are resolved against the process cwd.

    Returns a tuple of (media_type, base64_data).
    """
    if source.startswith(("http://", "https://")):
        resp = httpx.get(source, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        media_type = resp.headers.get("content-type", "image/png")
        raw = resp.content
    else:
        path = Path(source)
        if not path.is_absolute() and work_dir:
            path = Path(work_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        media_type = _image_media_type(path)
        raw = path.read_bytes()

    return media_type, base64.b64encode(raw).decode("utf-8")


def _image_data_url(source: str, timeout: float = DEFAULT_REQUEST_TIMEOUT) -> str:
    media_type, b64_data = _load_image(source, timeout=timeout)
    return f"data:{media_type};base64,{b64_data}"


def _image_reference(source: str, timeout: float = DEFAULT_REQUEST_TIMEOUT) -> dict:
    if source.startswith(("http://", "https://", "data:")):
        return {"image_url": source}
    return {"image_url": _image_data_url(source, timeout=timeout)}


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
def _open_images_for_edit(
    sources: list[str],
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    work_dir: str | None = None,
):
    """Open local paths as files and remote/data sources as in-memory files."""
    with ExitStack() as stack:
        files = []
        for index, source in enumerate(sources, start=1):
            if _is_local_image_source(source):
                path = Path(source)
                if not path.is_absolute() and work_dir:
                    path = Path(work_dir) / path
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {source}")
                files.append(stack.enter_context(path.open("rb")))
                continue

            media_type, b64_data = _load_image(source, timeout=timeout)
            buffer = BytesIO(base64.b64decode(b64_data))
            buffer.name = f"image-{index}.{_extension_for_media_type(media_type)}"
            files.append(buffer)

        yield files


@contextmanager
def _open_mask_for_edit(mask: str | None, timeout: float = DEFAULT_REQUEST_TIMEOUT, work_dir: str | None = None):
    if not mask:
        yield None
        return

    if _is_local_image_source(mask):
        path = Path(mask)
        if not path.is_absolute() and work_dir:
            path = Path(work_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask}")
        with path.open("rb") as handle:
            yield handle
        return

    media_type, b64_data = _load_image(mask, timeout=timeout)
    buffer = BytesIO(base64.b64decode(b64_data))
    buffer.name = f"mask.{_extension_for_media_type(media_type)}"
    yield buffer


def _image_model_uses_implicit_high_fidelity(image_model: str) -> bool:
    return image_model == "gpt-image-2"


# ── API callers ───────────────────────────────────────────────────────────

def _stream_text(config: ToolConfig, path: str, payload: dict, stream_type: str) -> str:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{config.base_url.rstrip('/')}/{path.lstrip('/')}"
    parts: list[str] = []

    with httpx.stream(
        "POST",
        url,
        headers=headers,
        json=payload,
        timeout=config.request_timeout,
    ) as response:
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


def _call_chat(config: ToolConfig, prompt: str, data_url: str, detail: str) -> str:
    return _stream_text(
        config,
        "chat/completions",
        {
            "model": config.model,
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


def _call_responses(config: ToolConfig, prompt: str, data_url: str, detail: str) -> str:
    return _stream_text(
        config,
        "responses",
        {
            "model": config.model,
            "input": _responses_payload(prompt, data_url, detail),
            "stream": True,
        },
        "responses",
    )


def _call_image_generation(config: ToolConfig, payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{config.base_url.rstrip('/')}/images/generations"

    with httpx.Client(timeout=config.request_timeout) as http_client:
        response = http_client.post(url, headers=headers, json=payload)

    if response.status_code >= 400:
        body = response.text
        raise RuntimeError(f"API error {response.status_code}: {body}")

    return response.json()


def _call_image_edit(config: ToolConfig, payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{config.base_url.rstrip('/')}/images/edits"

    with httpx.Client(timeout=config.request_timeout) as http_client:
        response = http_client.post(url, headers=headers, json=payload)

    if response.status_code >= 400:
        body = response.text
        raise RuntimeError(f"API error {response.status_code}: {body}")

    return response.json()


# ── MCP tool ─────────────────────────────────────────────────────────────

@mcp.tool(
    description="Analyse an image using the OpenAI vision API and return the text or description it contains. Local file paths support relative paths (resolved relative to work_dir); HTTP(S) URLs are unaffected."
)
def ocr_image(
    source: Annotated[str, Field(description='Local image path (resolved relative to work_dir) or HTTP(S) URL')],
    work_dir: Annotated[str, Field(description='Working directory for resolving relative local image paths.')],
    prompt: Annotated[str, Field(description='Custom instruction for the vision model.')] = "Please read and describe all the text and visual content in this image in detail.",
    detail: Annotated[str, Field(description='Image detail level: "auto", "low", or "high".')] = "auto",
    api_mode: Annotated[str | None, Field(description='API mode override: "chat" or "responses". Falls back to OPENAI_API_MODE env var, then "chat".')] = None,
) -> str:
    """Analyse an image using the OpenAI vision API."""
    config = _resolve_ocr_config()
    mode = _resolve_ocr_api_mode(api_mode)
    caller = {
        "chat": _call_chat,
        "responses": _call_responses,
    }.get(mode)
    if caller is None:
        raise ValueError(f"Unsupported API mode: {mode}")

    media_type, b64_data = _load_image(source, timeout=config.request_timeout, work_dir=work_dir)
    data_url = f"data:{media_type};base64,{b64_data}"
    return caller(config, prompt, data_url, detail)


@mcp.tool(
    description="Generate image files from a text prompt using the OpenAI image generation API."
)
def generate_image(
    prompt: Annotated[str, Field(description='Text prompt describing the image to generate.')],
    output_path: Annotated[str | None, Field(description='Optional output file path or directory. If omitted, images are saved under OPENAI_IMAGE_OUTPUT_DIR.')] = None,
    size: Annotated[str, Field(description='Image size, e.g. "1024x1024", "1024x1536", or "1536x1024".')] = "1024x1024",
    quality: Annotated[str, Field(description='Image quality: "auto", "low", "medium", or "high".')] = "auto",
    output_format: Annotated[str, Field(description='Output format: "png", "jpeg", or "webp".')] = "png",
    n: Annotated[int, Field(description='Number of images to generate (minimum 1).')] = 1,
    background: Annotated[str | None, Field(description='Optional background mode if supported by the image model.')] = None,
    user: Annotated[str | None, Field(description='Optional end-user identifier for API abuse monitoring.')] = None,
) -> str:
    """Generate images from a text prompt and save them locally."""
    if n < 1:
        raise ValueError("n must be at least 1.")

    config = _resolve_generate_image_config()
    payload = {
        "model": config.model,
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

    result = _call_image_generation(config, payload)
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
    source: Annotated[str | list[str], Field(description='Local file path, HTTP(S) URL, data URL, or list of image sources to edit.')],
    prompt: Annotated[str, Field(description='Text prompt describing the desired edit.')],
    work_dir: Annotated[str, Field(description='Working directory for resolving relative local paths for source and mask.')],
    mask: Annotated[str | None, Field(description='Optional local path, HTTP(S) URL, or data URL for an edit mask.')] = None,
    output_path: Annotated[str | None, Field(description='Optional output file path or directory. If omitted, images are saved under OPENAI_IMAGE_OUTPUT_DIR.')] = None,
    size: Annotated[str, Field(description='Output image size, e.g. "auto", "1024x1024", "1024x1536", or "1536x1024".')] = "auto",
    quality: Annotated[str, Field(description='Output quality: "auto", "low", "medium", or "high".')] = "auto",
    output_format: Annotated[str, Field(description='Output format: "png", "jpeg", or "webp".')] = "png",
    n: Annotated[int, Field(description='Number of edited images to generate (minimum 1).')] = 1,
    background: Annotated[str | None, Field(description='Optional background mode: "auto", "transparent", or "opaque".')] = None,
    input_fidelity: Annotated[str | None, Field(description='Optional fidelity level for the input image(s): "high" or "low".')] = None,
    moderation: Annotated[str | None, Field(description='Optional moderation level for GPT image models: "auto" or "low".')] = None,
    output_compression: Annotated[int | None, Field(description='Optional 0-100 compression level for JPEG/WebP.')] = None,
    user: Annotated[str | None, Field(description='Optional end-user identifier for API abuse monitoring.')] = None,
) -> str:
    """Edit one or more input images from a prompt and save the results locally."""
    if n < 1:
        raise ValueError("n must be at least 1.")
    if output_compression is not None and not 0 <= output_compression <= 100:
        raise ValueError("output_compression must be between 0 and 100.")

    config = _resolve_edit_image_config()
    request = {
        "model": config.model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "n": n,
    }
    if background:
        request["background"] = background
    if input_fidelity and not _image_model_uses_implicit_high_fidelity(config.model):
        request["input_fidelity"] = input_fidelity
    if moderation:
        request["moderation"] = moderation
    if output_compression is not None:
        request["output_compression"] = output_compression
    if user:
        request["user"] = user

    sources = _normalize_image_sources(source)
    with _open_images_for_edit(
        sources,
        timeout=config.request_timeout,
        work_dir=work_dir,
    ) as image_files, _open_mask_for_edit(
        mask,
        timeout=config.request_timeout,
        work_dir=work_dir,
    ) as mask_file:
        request["image"] = image_files if len(image_files) > 1 else image_files[0]
        if mask_file is not None:
            request["mask"] = mask_file
        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout,
        )
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
    for name in _disabled_tools():
        mcp.remove_tool(name)
    mcp.run()


if __name__ == "__main__":
    main()
