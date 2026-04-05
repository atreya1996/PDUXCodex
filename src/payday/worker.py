from __future__ import annotations

import logging
import time

from dotenv import load_dotenv

from payday.config import SettingsConfigurationError, get_settings
from payday.service import PaydayAppService

logger = logging.getLogger(__name__)
WORKER_SLEEP_SECONDS = 1.0


def run() -> int:
    """Run the background queue processor loop."""

    load_dotenv()
    get_settings.cache_clear()
    settings = get_settings()

    try:
        app_service = PaydayAppService(settings)
    except SettingsConfigurationError:
        logger.exception("Worker failed startup validation for runtime settings.")
        return 1

    logger.info("PayDay worker started.")
    try:
        while True:
            processed = app_service.run_worker_cycle(max_jobs=3)
            if processed == 0:
                time.sleep(WORKER_SLEEP_SECONDS)
    except KeyboardInterrupt:
        logger.info("PayDay worker received shutdown signal.")
    finally:
        app_service.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
