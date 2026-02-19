#!/usr/bin/env python3
"""Entry point for KittenTTS WebUI."""

import argparse
from webui.server import run_server


def main():
    parser = argparse.ArgumentParser(
        description="KittenTTS WebUI - A cute kitten-themed text-to-speech interface"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=7880, help="Port to bind to (default: 7880)"
    )
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
