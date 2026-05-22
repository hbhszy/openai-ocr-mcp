"""Test script that calls the ocr_image tool directly (bypasses MCP transport).

Usage:
    OPENAI_API_KEY=sk-xxx uv run python scripts/test_ocr.py <image_path_or_url> [prompt]

Examples:
    uv run python scripts/test_ocr.py screenshot.png
    uv run python scripts/test_ocr.py https://example.com/photo.jpg "提取图中所有文字"
    uv run python scripts/test_ocr.py ~/Desktop/test.png --detail high
    uv run python scripts/test_ocr.py ~/Desktop/test.png --api-mode responses
"""

import argparse
import os
import sys
from pathlib import Path

# ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if not os.environ.get("OPENAI_API_KEY"):
    # try loading .env
    from dotenv import load_dotenv
    load_dotenv()

from openai_ocr_mcp.server import ocr_image, _resolve_ocr_api_mode, _resolve_ocr_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the ocr_image tool")
    parser.add_argument("source", help="Local file path or HTTP(S) URL of the image")
    parser.add_argument("prompt", nargs="?", default=None,
                        help="Custom instruction for the vision model")
    parser.add_argument("--detail", choices=["auto", "low", "high"], default="auto",
                        help="Image detail level")
    parser.add_argument(
        "--api-mode",
        choices=["chat", "responses"],
        default=None,
        help="API mode override",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Print raw API response for debugging")
    args = parser.parse_args()

    api_mode = _resolve_ocr_api_mode(args.api_mode)
    try:
        config = _resolve_ocr_config()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    prompt = args.prompt or "Please read and describe all the text and visual content in this image in detail."

    print(f"Analyzing: {args.source}")
    print(f"Model:     {config.model}")
    print(f"Base URL:  {config.base_url}")
    print(f"Mode:      {api_mode}")
    print(f"Detail:    {args.detail}")
    print(f"Prompt:    {prompt}")
    print("-" * 60)

    if args.debug:
        print("Debug: using streaming request")

    result = ocr_image(source=args.source, prompt=prompt, detail=args.detail, api_mode=api_mode)

    print("\nResult:")
    print(result)


if __name__ == "__main__":
    main()
