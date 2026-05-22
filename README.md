# OpenAI OCR MCP

MCP server that uses OpenAI vision API to recognize and understand image content. Designed for non-multimodal AI models that need assistance interpreting images.

## Quick Start

```bash
# Install dependencies
uv sync

# Configure
cp .env.example .env
# edit .env with your API key

# Run
uv run openai-ocr-mcp
```

## Configuration

Configure via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL (compatible with other OpenAI-like providers) |
| `OPENAI_MODEL` | `gpt-5.4` | Model name |
| `OPENAI_API_MODE` | `chat` | API mode: `chat` (Chat Completions) or `responses` (Responses API) |
| `OPENAI_OCR_MCP_CONFIG` | `./config.json` if present | Optional JSON config file path for structured defaults and per-tool overrides |
| `OPENAI_IMAGE_MODEL` | `gpt-image-2` | Image generation/editing model name |
| `OPENAI_IMAGE_OUTPUT_DIR` | `generated_images` | Default directory for generated or edited image files |
| `OPENAI_REQUEST_TIMEOUT` | `1200` | Request timeout in seconds for OCR and image APIs |
| `OPENAI_IMAGE_REQUEST_TIMEOUT` | `1200` | Legacy image generation/editing request timeout override |

Tool-specific environment variables are also supported:

| Tool | API key | Base URL | Model | Extra |
|---|---|---|---|---|
| `ocr_image` | `OPENAI_OCR_API_KEY` | `OPENAI_OCR_BASE_URL` | `OPENAI_OCR_MODEL` | `OPENAI_OCR_API_MODE`, `OPENAI_OCR_REQUEST_TIMEOUT` |
| `generate_image` | `OPENAI_GENERATE_IMAGE_API_KEY` | `OPENAI_GENERATE_IMAGE_BASE_URL` | `OPENAI_GENERATE_IMAGE_MODEL` | `OPENAI_GENERATE_IMAGE_REQUEST_TIMEOUT` |
| `edit_image` | `OPENAI_EDIT_IMAGE_API_KEY` | `OPENAI_EDIT_IMAGE_BASE_URL` | `OPENAI_EDIT_IMAGE_MODEL` | `OPENAI_EDIT_IMAGE_REQUEST_TIMEOUT` |

`OPENAI_OCR_IMAGE_*` is also accepted as an alias for `OPENAI_OCR_*`.

### Structured config file

For larger setups, create `config.json` in the working directory, or set `OPENAI_OCR_MCP_CONFIG` to another JSON file. A complete example is included at `config.example.json`:

```json
{
  "defaults": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxx",
    "request_timeout": 1200
  },
  "tools": {
    "ocr_image": {
      "base_url": "",
      "api_key": "",
      "model": "",
      "api_mode": "chat",
      "request_timeout": 1200
    },
    "generate_image": {
      "base_url": "",
      "api_key": "",
      "model": "",
      "request_timeout": 1200
    },
    "edit_image": {
      "base_url": "",
      "api_key": "",
      "model": "",
      "request_timeout": 1200
    }
  }
}
```

Each tool can define `base_url`, `api_key`, `model`, and `request_timeout`. `ocr_image` can also define `api_mode`. Empty string values are ignored, so the tool falls back to environment variables, `defaults`, or the tool's built-in model default.

If you prefer to keep secrets outside JSON, use `api_key_env` instead of `api_key`:

```json
{
  "defaults": {
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

Per-field resolution order:

| Field | Priority |
|---|---|
| API key | Tool-specific env → tool config `api_key`/`api_key_env` → `OPENAI_API_KEY` → defaults config `api_key`/`api_key_env` |
| Base URL | Tool-specific env → tool config → `OPENAI_BASE_URL` → defaults config → `https://api.openai.com/v1` |
| OCR model | `OPENAI_OCR_MODEL` → tool config → `OPENAI_MODEL` → defaults config → `gpt-5.4` |
| Image model | Tool-specific env → tool config → `OPENAI_IMAGE_MODEL` → `OPENAI_MODEL` → defaults config → `gpt-image-2` |
| Request timeout | Tool-specific env → `OPENAI_REQUEST_TIMEOUT` → `OPENAI_IMAGE_REQUEST_TIMEOUT` for image tools → tool config `request_timeout` → defaults config `request_timeout` → `1200` |
| OCR API mode | tool parameter → tool-specific env → tool config → `OPENAI_API_MODE` → defaults config → `chat` |

### Priority (highest → lowest)

| Priority | Source | Example |
|---|---|---|
| 1 | MCP client `env` field | `"env": {"OPENAI_API_KEY": "..."}` in `mcpServers` config |
| 2 | Shell environment variables | `export OPENAI_API_KEY=...` |
| 3 | `.env` file | `OPENAI_API_KEY=...` in project root |

In other words: if an environment variable is set in the MCP client config, it wins over the shell and `.env` for that same variable. Structured config then applies according to the per-field rules above.

## Tools

### `ocr_image`

Analyse an image and return its text/visual content.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `string` | — | Local file path or HTTP(S) URL of the image |
| `prompt` | `string` | *(see below)* | Custom instruction for the vision model |
| `detail` | `string` | `"auto"` | Image detail level: `auto`, `low`, or `high` |
| `api_mode` | `string` | `null` | API mode override (`"chat"` or `"responses"`); falls back to `OPENAI_API_MODE` env var, then `"chat"` |

The OCR request timeout defaults to 1200 seconds and can be configured with `OPENAI_OCR_REQUEST_TIMEOUT`, `OPENAI_REQUEST_TIMEOUT`, or `request_timeout` in the structured config file.

Default prompt: `"Please read and describe all the text and visual content in this image in detail."`

### `generate_image`

Generate image files from a text prompt using the OpenAI image generation API.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | - | Text prompt describing the image to generate |
| `output_path` | `string` or `null` | `null` | Optional output file path or directory. If omitted, images are saved under `OPENAI_IMAGE_OUTPUT_DIR` |
| `size` | `string` | `1024x1024` | Image size, such as `1024x1024`, `1024x1536`, or `1536x1024` |
| `quality` | `string` | `auto` | Image quality, typically `auto`, `low`, `medium`, or `high` |
| `output_format` | `string` | `png` | Output format, typically `png`, `jpeg`, or `webp` |
| `n` | `integer` | `1` | Number of images to generate |
| `background` | `string` or `null` | `null` | Optional background mode if supported by the image model |
| `user` | `string` or `null` | `null` | Optional end-user identifier for API abuse monitoring |

The image model is configured through `OPENAI_GENERATE_IMAGE_MODEL`, the structured config file, or the legacy shared `OPENAI_IMAGE_MODEL`; it is not exposed as a tool parameter. The image request timeout defaults to 1200 seconds and can be configured with `OPENAI_GENERATE_IMAGE_REQUEST_TIMEOUT`, `OPENAI_IMAGE_REQUEST_TIMEOUT`, or `request_timeout` in the structured config file. The tool returns JSON with the saved local file paths and any API-provided revised prompts or usage data.

### `edit_image`

Edit existing image files from a text prompt using the OpenAI image editing API.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `source` | `string` or `array[string]` | - | Local file path, HTTP(S) URL, data URL, or list of image sources to edit |
| `prompt` | `string` | - | Text prompt describing the desired edit |
| `mask` | `string` or `null` | `null` | Optional local file path, HTTP(S) URL, or data URL for an edit mask |
| `output_path` | `string` or `null` | `null` | Optional output file path or directory. If omitted, images are saved under `OPENAI_IMAGE_OUTPUT_DIR` |
| `size` | `string` | `auto` | Output image size, such as `auto`, `1024x1024`, `1024x1536`, or `1536x1024` |
| `quality` | `string` | `auto` | Output quality, typically `auto`, `low`, `medium`, or `high` |
| `output_format` | `string` | `png` | Output format, typically `png`, `jpeg`, or `webp` |
| `n` | `integer` | `1` | Number of edited images to generate |
| `background` | `string` or `null` | `null` | Optional background mode, such as `auto`, `transparent`, or `opaque` |
| `input_fidelity` | `string` or `null` | `null` | Optional input fidelity level, typically `high` or `low` |
| `moderation` | `string` or `null` | `null` | Optional moderation level, typically `auto` or `low` |
| `output_compression` | `integer` or `null` | `null` | Optional 0-100 compression level for `jpeg` or `webp` output |
| `user` | `string` or `null` | `null` | Optional end-user identifier for API abuse monitoring |

The image model is configured through `OPENAI_EDIT_IMAGE_MODEL`, the structured config file, or the legacy shared `OPENAI_IMAGE_MODEL`. The image request timeout defaults to 1200 seconds and can be configured with `OPENAI_EDIT_IMAGE_REQUEST_TIMEOUT`, `OPENAI_IMAGE_REQUEST_TIMEOUT`, or `request_timeout` in the structured config file. The tool returns JSON with the saved local file paths and any API-provided revised prompts or usage data.

## Using with MCP Clients

### Claude Desktop / Cline / etc.

```json
{
  "mcpServers": {
    "openai-ocr": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/openai-ocr-mcp", "openai-ocr-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-xxx",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_MODEL": "gpt-5.4",
        "OPENAI_IMAGE_MODEL": "gpt-image-2",
        "OPENAI_OCR_MCP_CONFIG": "/path/to/openai-ocr-mcp/config.json"
      }
    }
  }
}
```

Replace `/path/to/openai-ocr-mcp` with the actual project path.

## Testing

A test script is provided to call the OCR function directly (bypasses MCP transport) for quick verification:

```bash
# Analyze a local image
OPENAI_API_KEY=sk-xxx uv run python scripts/test_ocr.py ~/Desktop/screenshot.png

# Analyze a remote image
uv run python scripts/test_ocr.py https://example.com/photo.jpg

# Custom prompt (e.g. extract text in Chinese)
uv run python scripts/test_ocr.py receipt.jpg "提取图中所有文字"

# Control image detail level
uv run python scripts/test_ocr.py diagram.png --detail high

# Use Responses API
uv run python scripts/test_ocr.py screenshot.png --api-mode responses
```

The implementation uses streaming requests internally and returns the assembled
final text.

The script loads `.env` automatically if `OPENAI_API_KEY` is not already set.
