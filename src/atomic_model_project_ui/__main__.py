import argparse
from pathlib import Path

from atomic_model_project_ui.server import create_server


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Local EOS parameter editor at 127.0.0.1; trusted single-user workstation"
    )
    p.add_argument("store", type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--exports", type=Path)
    p.add_argument("--study", action="store_true", help="Open a model/data study store")
    p.add_argument("--rss-limit-mib", type=int, default=1024)
    args = p.parse_args(argv)
    backend = None
    try:
        if args.study:
            from atomic_model_results.backend import Backend

            backend = Backend(args.store, rss_limit_bytes=args.rss_limit_mib * 1024**2)
        server = create_server(args.store, port=args.port, exports=args.exports, backend=backend)
    except (ValueError, OSError) as exc:
        if backend is not None:
            backend.close()
        p.error(str(exc))
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if backend is not None:
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
