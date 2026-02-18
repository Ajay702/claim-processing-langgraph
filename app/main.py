"""FastAPI application entrypoint for the Claim Processing Pipeline."""

import logging
import os
import sys

import uvicorn
from fastapi import FastAPI

from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Claim Processing Pipeline",
    description="Production-ready API for processing insurance claim documents.",
    version="0.1.0",
)

app.include_router(router)


def main() -> None:
    """Launch the Uvicorn server with configuration from environment variables."""
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server on port %d", port)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
