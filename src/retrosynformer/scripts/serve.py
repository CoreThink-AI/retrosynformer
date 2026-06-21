"""CLI entry point for the RetroSynFormer inference server.

Usage::

    rs-serve [--host HOST] [--port PORT] [--workers N]

Required environment variables (set before running):

    MODEL_CONFIG_PATH      Path to config.yaml
    MODEL_WEIGHTS_PATH     Path to model.pth
    BUILDING_BLOCKS_PATH   Path to building_blocks CSV
    TEMPLATES_PATH         Path to reaction templates pickle
    API_KEY                Secret for X-API-Key header auth
"""

import argparse
import os
from retrosynformer.scripts import print_banner


def main() -> None:
    print_banner()
    parser = argparse.ArgumentParser(
        description="Start the RetroSynFormer FastAPI inference server."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8080)),
        help="Bind port (default: $PORT or 8080)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1; GPU builds must use 1)",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is not installed. Run: uv sync --extra serve"
        ) from exc

    uvicorn.run(
        "retrosynformer.serve.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
