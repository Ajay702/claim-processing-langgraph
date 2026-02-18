"""API route definitions for the claim processing pipeline."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES: set[str] = {"application/pdf"}
ALLOWED_EXTENSIONS: set[str] = {".pdf"}


def _validate_claim_id(claim_id: str) -> str:
    """Validate that claim_id is a non-empty string.

    Args:
        claim_id: The claim identifier to validate.

    Returns:
        The stripped claim_id.

    Raises:
        HTTPException: If claim_id is empty or whitespace-only.
    """
    stripped = claim_id.strip()
    if not stripped:
        logger.warning("Received empty claim_id")
        raise HTTPException(status_code=400, detail="claim_id must not be empty.")
    return stripped


def _validate_pdf(file: UploadFile) -> None:
    """Validate that the uploaded file is a PDF.

    Args:
        file: The uploaded file to validate.

    Raises:
        HTTPException: If the file is not a valid PDF.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            "Invalid content type: %s for file: %s", file.content_type, file.filename
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only PDF files are accepted.",
        )

    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        logger.warning("Invalid file extension: %s", extension)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{extension}'. Only .pdf files are accepted.",
        )


async def _save_to_tmp(file: UploadFile) -> Path:
    """Persist the uploaded file to a temporary directory.

    Args:
        file: The uploaded file to save.

    Returns:
        The path to the saved temporary file.

    Raises:
        HTTPException: If the file cannot be saved.
    """
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="claim_"))
        destination = tmp_dir / (file.filename or "upload.pdf")
        content = await file.read()
        destination.write_bytes(content)
        logger.info("Saved uploaded file to %s (%d bytes)", destination, len(content))
        return destination
    except Exception as exc:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(
            status_code=500, detail="Failed to save uploaded file."
        ) from exc


@router.post("/api/process", status_code=200)
async def process_claim(
    claim_id: str = Form(..., description="Unique claim identifier"),
    file: UploadFile = File(..., description="PDF document to process"),
) -> dict[str, Any]:
    """Accept a claim PDF for processing.

    Validates the claim_id and uploaded file, persists the file to a
    temporary location, and returns an acknowledgement response.

    Args:
        claim_id: Unique identifier for the claim.
        file: PDF file to be processed.

    Returns:
        A dict containing the claim_id, filename, and processing status.
    """
    validated_claim_id = _validate_claim_id(claim_id)
    _validate_pdf(file)
    saved_path = await _save_to_tmp(file)
    logger.info(
        "Claim received — claim_id=%s file=%s path=%s",
        validated_claim_id,
        file.filename,
        saved_path,
    )
    return {
        "claim_id": validated_claim_id,
        "filename": file.filename,
        "status": "received",
    }


@router.get("/health", status_code=200)
async def health_check() -> dict[str, str]:
    """Liveness / readiness health check endpoint.

    Returns:
        A dict with the current service status.
    """
    return {"status": "ok"}
