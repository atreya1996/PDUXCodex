from __future__ import annotations

import logging
import os

from payday.api.server import build_server


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("PAYDAY_BACKEND_API_HOST", "127.0.0.1")
    port = int(os.getenv("PAYDAY_BACKEND_API_PORT", "8000"))
    server = build_server(host=host, port=port)
    logging.getLogger(__name__).info("PayDay backend listening on %s:%s", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
