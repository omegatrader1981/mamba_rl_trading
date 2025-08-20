# src/utils.py
# <<< REFACTORED: Contains only general-purpose, non-framework-specific helpers. >>>

import logging
import os

log = logging.getLogger(__name__)

def ensure_dir(directory_path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if directory_path:
        try:
            os.makedirs(directory_path, exist_ok=True)
        except OSError as e:
            log.error(f"Failed to create directory {directory_path}: {e}")
            raise

def format_currency(value: float) -> str:
    """Formats a float value as a currency string (e.g., $12,345.67)."""
    try:
        return f"${value:,.2f}"
    except (TypeError, ValueError):
        log.warning(f"Could not format value '{value}' as currency. Returning as string.")
        return str(value)