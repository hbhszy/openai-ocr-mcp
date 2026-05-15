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
| `OPENAI_MODEL` | `gpt-4o` | Model name |
| `OPENAI_API_MODE` | `chat` | API mode: `chat` (Chat Completions) or `responses` (Responses API) |
| `OPENAI_IMAGE_MODEL` | `gpt-image-2` | Image generation model name |
| `OPENAI_IMAGE_OUTPUT_DIR` | `generated_images` | Default directory for generated image files |

### Priority (highest → lowest)

| Priority | Source | Example |
|---|---|---|
| 1 | MCP client `env` field | `"env": {"OPENAI_API_KEY": "..."}` in `mcpServers` config |
| 2 | Shell environment variables | `export OPENAI_API_KEY=...` |
| 3 | `.env` file | `OPENAI_API_KEY=...` in project root |

In other words: if `OPENAI_API_KEY` is set in the MCP client config, it wins over everything. Otherwise the shell's env var wins, and finally `.env` is used as fallback.

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

Default prompt: `"Please read and describe all the text and visual content in this image in detail."`

### `generate_image`

Generate image files from a text prompt using the OpenAI image generation API.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | - | Text prompt describing the image to generate |
| `output_path` | `string` or `null` | `null` | Optional output file path or directory. If omitted, images are saved under `OPENAI_IMAGE_OUTPUT_DIR` |
| `model` | `string` or `null` | `null` | Image model override; falls back to `OPENAI_IMAGE_MODEL`, then `gpt-image-2` |
| `size` | `string` | `1024x1024` | Image size, such as `1024x1024`, `1024x1536`, or `1536x1024` |
| `quality` | `string` | `auto` | Image quality, typically `auto`, `low`, `medium`, or `high` |
| `output_format` | `string` | `png` | Output format, typically `png`, `jpeg`, or `webp` |
| `n` | `integer` | `1` | Number of images to generate |
| `background` | `string` or `null` | `null` | Optional background mode if supported by the image model |
| `user` | `string` or `null` | `null` | Optional end-user identifier for API abuse monitoring |

The tool returns JSON with the saved local file paths and any API-provided revised prompts or usage data.

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
        "OPENAI_MODEL": "gpt-4o",
        "OPENAI_IMAGE_MODEL": "gpt-image-2"
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
