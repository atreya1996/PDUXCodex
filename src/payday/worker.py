from __future__ import annotations

import logging
import time

from payday.config import get_settings
from payday.service import PaydayAppService

logger = logging.getLogger(__name__)
DEFAULT_POLL_INTERVAL_SECONDS = 1.0


def run_worker_forever(*, poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS, max_jobs_per_cycle: int = 1) -> None:
    settings = get_settings()
    service = PaydayAppService(settings)
    logger.info("PayDay worker started with poll_interval_seconds=%s", poll_interval_seconds)
    while True:
        processed = service.run_worker_cycle(max_jobs=max_jobs_per_cycle)
        if processed == 0:
            time.sleep(poll_interval_seconds)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_worker_forever()


if __name__ == "__main__":
    main()
