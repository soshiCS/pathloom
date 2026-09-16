"""Run the MemberOps Sandbox:  python -m examples.member_ops --port 8765 --mode normal"""
from __future__ import annotations

import argparse

from .app import MODES, serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.member_ops", description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mode", choices=MODES, default="normal",
                        help="runtime state to demonstrate (default: normal)")
    args = parser.parse_args(argv)
    server = serve(port=args.port, mode=args.mode, host=args.host)
    print(f"MemberOps Sandbox ({args.mode}) at http://{args.host}:{args.port}/  "
          f"sign in as 'operator' with the training password; POST /__reset clears state; Ctrl+C stops")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
