# OpenAI OCR MCP

基于 OpenAI 视觉 API 的 MCP 服务器，用于识别和理解图片内容。专为需要图像理解能力的非多模态 AI 模型设计。

## 快速开始

```bash
# 安装依赖
uv sync

# 配置
cp .env.example .env
# 编辑 .env 填入 API Key

# 运行
uv run openai-ocr-mcp
```

## 配置

通过环境变量或 `.env` 文件配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API 密钥（必填） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 基础地址（兼容其他 OpenAI 兼容服务商） |
| `OPENAI_MODEL` | `gpt-5.4` | 模型名称 |
| `OPENAI_API_MODE` | `chat` | API 模式：`chat`（Chat Completions）或 `responses`（Responses API） |
| `OPENAI_OCR_MCP_CONFIG` | `./config.json`（如存在） | JSON 配置文件路径，用于结构化默认值和工具级覆盖 |
| `OPENAI_OCR_MCP_DISABLED_TOOLS` | — | 逗号分隔的工具名称，对 MCP 客户端隐藏（如 `generate_image,edit_image`） |
| `OPENAI_IMAGE_MODEL` | `gpt-image-2` | 图片生成/编辑模型名称 |
| `OPENAI_IMAGE_OUTPUT_DIR` | `generated_images` | 生成图片的默认保存目录 |
| `OPENAI_REQUEST_TIMEOUT` | `1200` | OCR 和图片 API 请求超时（秒） |
| `OPENAI_IMAGE_REQUEST_TIMEOUT` | `1200` | 旧版图片生成/编辑请求超时覆盖 |

也支持工具级别的环境变量：

| 工具 | API Key | Base URL | 模型 | 其他 |
|---|---|---|---|---|
| `ocr_image` | `OPENAI_OCR_API_KEY` | `OPENAI_OCR_BASE_URL` | `OPENAI_OCR_MODEL` | `OPENAI_OCR_API_MODE`、`OPENAI_OCR_REQUEST_TIMEOUT` |
| `generate_image` | `OPENAI_GENERATE_IMAGE_API_KEY` | `OPENAI_GENERATE_IMAGE_BASE_URL` | `OPENAI_GENERATE_IMAGE_MODEL` | `OPENAI_GENERATE_IMAGE_REQUEST_TIMEOUT` |
| `edit_image` | `OPENAI_EDIT_IMAGE_API_KEY` | `OPENAI_EDIT_IMAGE_BASE_URL` | `OPENAI_EDIT_IMAGE_MODEL` | `OPENAI_EDIT_IMAGE_REQUEST_TIMEOUT` |

`OPENAI_OCR_IMAGE_*` 也可作为 `OPENAI_OCR_*` 的别名。

### 结构化配置文件

对于更复杂的配置，可在工作目录创建 `config.json`，或通过 `OPENAI_OCR_MCP_CONFIG` 指定其他 JSON 文件。完整示例见 `config.example.json`：

```json
{
  "defaults": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxx",
    "request_timeout": 1200
  },
  "disabled_tools": [],
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

每个工具可定义 `base_url`、`api_key`、`model`、`request_timeout` 和 `enabled`。`ocr_image` 还可定义 `api_mode`。空字符串值会被忽略，工具会回退到环境变量、`defaults` 或内置默认值。

如果希望密钥不存储在 JSON 中，可使用 `api_key_env` 代替 `api_key`：

```json
{
  "defaults": {
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

### 禁用工具

可以对 MCP 客户端隐藏工具，使其不出现在工具列表中。当只需部分工具时非常有用。

支持三种方式（可组合使用）：

1. **配置文件顶层 `disabled_tools` 列表**：

```json
{
  "disabled_tools": ["generate_image", "edit_image"]
}
```

2. **工具级别 `enabled: false`**：

```json
{
  "tools": {
    "edit_image": {
      "enabled": false
    }
  }
}
```

3. **环境变量** `OPENAI_OCR_MCP_DISABLED_TOOLS`（逗号分隔）：

```bash
OPENAI_OCR_MCP_DISABLED_TOOLS=generate_image,edit_image
```

三种来源合并生效。工具级别的 `enabled: true` 可覆盖 `disabled_tools` 中的禁用；环境变量优先级最高，不可被配置文件覆盖。

各字段解析顺序：

| 字段 | 优先级 |
|---|---|
| API Key | 工具级环境变量 → 工具配置 `api_key`/`api_key_env` → `OPENAI_API_KEY` → 默认配置 `api_key`/`api_key_env` |
| Base URL | 工具级环境变量 → 工具配置 → `OPENAI_BASE_URL` → 默认配置 → `https://api.openai.com/v1` |
| OCR 模型 | `OPENAI_OCR_MODEL` → 工具配置 → `OPENAI_MODEL` → 默认配置 → `gpt-5.4` |
| 图片模型 | 工具级环境变量 → 工具配置 → `OPENAI_IMAGE_MODEL` → `OPENAI_MODEL` → 默认配置 → `gpt-image-2` |
| 请求超时 | 工具级环境变量 → `OPENAI_REQUEST_TIMEOUT` → 图片工具的 `OPENAI_IMAGE_REQUEST_TIMEOUT` → 工具配置 `request_timeout` → 默认配置 `request_timeout` → `1200` |
| OCR API 模式 | 工具参数 → 工具级环境变量 → 工具配置 → `OPENAI_API_MODE` → 默认配置 → `chat` |

### 优先级（从高到低）

| 优先级 | 来源 | 示例 |
|---|---|---|
| 1 | MCP 客户端 `env` 字段 | `mcpServers` 配置中的 `"env": {"OPENAI_API_KEY": "..."}` |
| 2 | Shell 环境变量 | `export OPENAI_API_KEY=...` |
| 3 | `.env` 文件 | 项目根目录中的 `OPENAI_API_KEY=...` |

即：MCP 客户端配置中的环境变量优先于 Shell 和 `.env` 中的同名变量。结构化配置按上述各字段规则生效。

## 工具

### `ocr_image`

分析图片并返回文字/视觉内容。

**参数：**

| 名称 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `source` | `string` | — | 图片的本地文件路径或 HTTP(S) URL |
| `prompt` | `string` | *（见下方）* | 自定义视觉模型指令 |
| `detail` | `string` | `"auto"` | 图片细节级别：`auto`、`low` 或 `high` |
| `api_mode` | `string` | `null` | API 模式覆盖（`"chat"` 或 `"responses"`）；回退到 `OPENAI_API_MODE` 环境变量，然后默认 `"chat"` |

OCR 请求超时默认 1200 秒，可通过 `OPENAI_OCR_REQUEST_TIMEOUT`、`OPENAI_REQUEST_TIMEOUT` 或配置文件中的 `request_timeout` 配置。

默认 prompt：`"Please read and describe all the text and visual content in this image in detail."`

### `generate_image`

使用 OpenAI 图片生成 API 从文本提示生成图片文件。

**参数：**

| 名称 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | `string` | - | 描述要生成图片的文本提示 |
| `output_path` | `string` 或 `null` | `null` | 可选的输出文件路径或目录。省略时保存到 `OPENAI_IMAGE_OUTPUT_DIR` |
| `size` | `string` | `1024x1024` | 图片尺寸，如 `1024x1024`、`1024x1536` 或 `1536x1024` |
| `quality` | `string` | `auto` | 图片质量，通常为 `auto`、`low`、`medium` 或 `high` |
| `output_format` | `string` | `png` | 输出格式，通常为 `png`、`jpeg` 或 `webp` |
| `n` | `integer` | `1` | 生成图片数量 |
| `background` | `string` 或 `null` | `null` | 可选的背景模式（需模型支持） |
| `user` | `string` 或 `null` | `null` | 可选的终端用户标识，用于 API 滥用监控 |

图片模型通过 `OPENAI_GENERATE_IMAGE_MODEL`、结构化配置文件或旧版共享 `OPENAI_IMAGE_MODEL` 配置，不作为工具参数暴露。图片请求超时默认 1200 秒，可通过 `OPENAI_GENERATE_IMAGE_REQUEST_TIMEOUT`、`OPENAI_IMAGE_REQUEST_TIMEOUT` 或配置文件中的 `request_timeout` 配置。工具返回 JSON，包含保存的本地文件路径及 API 返回的修订提示词和使用量数据。

### `edit_image`

使用 OpenAI 图片编辑 API 从文本提示编辑现有图片。

**参数：**

| 名称 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `source` | `string` 或 `array[string]` | - | 本地文件路径、HTTP(S) URL、data URL 或图片来源列表 |
| `prompt` | `string` | - | 描述编辑内容的文本提示 |
| `mask` | `string` 或 `null` | `null` | 可选的编辑遮罩，支持本地路径、HTTP(S) URL 或 data URL |
| `output_path` | `string` 或 `null` | `null` | 可选的输出文件路径或目录。省略时保存到 `OPENAI_IMAGE_OUTPUT_DIR` |
| `size` | `string` | `auto` | 输出图片尺寸，如 `auto`、`1024x1024`、`1024x1536` 或 `1536x1024` |
| `quality` | `string` | `auto` | 输出质量，通常为 `auto`、`low`、`medium` 或 `high` |
| `output_format` | `string` | `png` | 输出格式，通常为 `png`、`jpeg` 或 `webp` |
| `n` | `integer` | `1` | 生成编辑图片数量 |
| `background` | `string` 或 `null` | `null` | 可选的背景模式，如 `auto`、`transparent` 或 `opaque` |
| `input_fidelity` | `string` 或 `null` | `null` | 可选的输入保真度，通常为 `high` 或 `low` |
| `moderation` | `string` 或 `null` | `null` | 可选的审核级别，通常为 `auto` 或 `low` |
| `output_compression` | `integer` 或 `null` | `null` | 可选的压缩级别（0-100），适用于 `jpeg` 或 `webp` 输出 |
| `user` | `string` 或 `null` | `null` | 可选的终端用户标识，用于 API 滥用监控 |

图片模型通过 `OPENAI_EDIT_IMAGE_MODEL`、结构化配置文件或旧版共享 `OPENAI_IMAGE_MODEL` 配置。图片请求超时默认 1200 秒，可通过 `OPENAI_EDIT_IMAGE_REQUEST_TIMEOUT`、`OPENAI_IMAGE_REQUEST_TIMEOUT` 或配置文件中的 `request_timeout` 配置。工具返回 JSON，包含保存的本地文件路径及 API 返回的修订提示词和使用量数据。

## 在 MCP 客户端中使用

### Claude Desktop / Cline 等

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

将 `/path/to/openai-ocr-mcp` 替换为实际项目路径。

## 测试

提供了测试脚本可直接调用 OCR 功能（绕过 MCP 传输层）进行快速验证：

```bash
# 分析本地图片
OPENAI_API_KEY=sk-xxx uv run python scripts/test_ocr.py ~/Desktop/screenshot.png

# 分析远程图片
uv run python scripts/test_ocr.py https://example.com/photo.jpg

# 自定义提示词（如提取中文文字）
uv run python scripts/test_ocr.py receipt.jpg "提取图中所有文字"

# 控制图片细节级别
uv run python scripts/test_ocr.py diagram.png --detail high

# 使用 Responses API
uv run python scripts/test_ocr.py screenshot.png --api-mode responses
```

内部使用流式请求，返回拼接后的完整文本。

如果 `OPENAI_API_KEY` 未设置，脚本会自动加载 `.env` 文件。
