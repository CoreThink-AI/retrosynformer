#!/usr/bin/env python
"""Launch the RetroSynFormer training dashboard.

Requires:  uv sync --extra dashboard

Usage
-----
    rs-dashboard
    rs-dashboard --port 5050 --no-sync --debug
    rs-dashboard --cloud-run-url https://retrosynformer-inference-xxx.run.app
    CLOUD_RUN_URL=https://... rs-dashboard
"""
import argparse
import sys


def main() -> None:
    try:
        from retrosynformer.dashboard import create_app
    except ImportError as e:
        sys.exit(
            f"Dashboard dependencies not installed: {e}\n"
            "Run: uv sync --extra dashboard"
        )

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--results", default="results/", dest="results_root",
                        help="Local results directory (default: results/)")
    parser.add_argument("--db", default=None, dest="db_url",
                        help="SQLAlchemy DB URL (default: sqlite:///<results>/dashboard.db)")
    parser.add_argument("--cloud-run-url", default=None, dest="cloud_run_url",
                        help="Cloud Run service URL for health panel")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip initial sync on startup")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(
        results_root=args.results_root,
        db_url=args.db_url,
        initial_sync=not args.no_sync,
        cloud_run_url=args.cloud_run_url,
        debug=args.debug,
    )

    print(f"Dashboard: http://{args.host}:{args.port}/")
    print(f"Admin:     http://{args.host}:{args.port}/admin/")
    print(f"API:       http://{args.host}:{args.port}/api/v1/studies")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
