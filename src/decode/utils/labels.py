from __future__ import annotations

import re


def clean_class_label(class_label: str) -> str:
    """Normalize class labels for display inside prompts and logs."""
    cleaned = re.sub(r"^\d+\.", "", class_label)
    cleaned = cleaned.replace("_", " ")
    return cleaned.strip()
