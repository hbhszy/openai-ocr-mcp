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

## Tool

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
        "OPENAI_MODEL": "gpt-4o"
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
