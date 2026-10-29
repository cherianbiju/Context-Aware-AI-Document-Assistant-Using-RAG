
from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from multi_doc_chat.logger.cutom_logger import CustomLogger
from multi_doc_chat.exception.custom_exception import DocumentPortalException


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".pptx",
    ".md",
    ".csv",
    ".xlsx",
    ".xls",
    ".db",
    ".sqlite",
    ".sqlite3",
}

# Local logger instance
log = CustomLogger().get_logger(__name__)


def save_uploaded_files(
    uploaded_files: Iterable,
    target_dir: Path,
) -> List[Path]:
    """Save uploaded files with their original names and timestamps."""

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for uf in uploaded_files:
            # Get the original filename
            name = (
                getattr(uf, "filename", None)
                or getattr(uf, "name", "file")
            )

            original_path = Path(name)
            ext = original_path.suffix.lower()

            # Skip unsupported file types
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning(
                    "Unsupported file skipped",
                    filename=name,
                )
                continue

            # Clean the original filename
            safe_name = re.sub(
                r"[^a-zA-Z0-9_-]",
                "_",
                original_path.stem,
            ).strip("_").lower()

            if not safe_name:
                safe_name = "file"

            # Generate upload timestamp
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            # Preserve the name, timestamp, and unique suffix
            unique_id = uuid.uuid4().hex[:6]

            fname = (
                f"{safe_name}_{timestamp}_{unique_id}{ext}"
            )

            out = target_dir / fname

            # Read the uploaded file content
            if hasattr(uf, "file") and hasattr(uf.file, "read"):
                uf.file.seek(0)
                data = uf.file.read()

            elif hasattr(uf, "read"):
                data = uf.read()

            else:
                buf = getattr(uf, "getbuffer", None)

                if callable(buf):
                    data = buf()
                else:
                    raise ValueError(
                        "Unsupported uploaded file object; "
                        "no readable interface"
                    )

            # Convert memoryview to bytes if necessary
            if isinstance(data, memoryview):
                data = data.tobytes()

            # Write the file into the session directory
            with open(out, "wb") as f:
                f.write(data)

            saved.append(out)

            log.info(
                "File saved for ingestion",
                uploaded=name,
                saved_as=str(out),
            )

        return saved

    except Exception as e:
        log.error(
            "Failed to save uploaded files",
            error=str(e),
            dir=str(target_dir),
        )

        raise DocumentPortalException(
            "Failed to save uploaded files",
            e,
        ) from e
