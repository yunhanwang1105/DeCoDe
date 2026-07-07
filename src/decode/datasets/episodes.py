from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from decode.config import DatasetConfig


@dataclass(frozen=True)
class Episode:
    sampled_classes: list[str]
    support_images: list[Path]
    query_images: list[Path]


def load_episodes(dataset: DatasetConfig, num_ways: int, num_episodes: int) -> list[Episode]:
    if not dataset.sampling_log.exists():
        raise FileNotFoundError(
            f"Sampling log not found: {dataset.sampling_log}\n"
            "Put episode JSON files under episode_logs/<dataset>/ or update the YAML config."
        )

    with dataset.sampling_log.open("r", encoding="utf-8") as f:
        sampling_log = json.load(f)

    episodes = []
    for entry in sampling_log[:num_episodes]:
        classes = [c.lstrip("/").strip() for c in entry["sampled_classes"][:num_ways]]
        support = [
            resolve_image_path(dataset.images_path, image_name, classes[i], dataset.resolve_mode)
            for i, image_name in enumerate(entry[dataset.support_key][:num_ways])
        ]
        query = [
            resolve_image_path(dataset.images_path, image_name, classes[i], dataset.resolve_mode)
            for i, image_name in enumerate(entry[dataset.query_key][:num_ways])
        ]
        _validate_paths(support, "support")
        _validate_paths(query, "query")
        episodes.append(Episode(sampled_classes=classes, support_images=support, query_images=query))

    return episodes


def resolve_image_path(images_path: Path, image_name: str, class_dir: str, resolve_mode: str) -> Path:
    image_name = image_name.lstrip("/").strip()
    class_dir = class_dir.lstrip("/").strip()

    if resolve_mode == "ucf_frame":
        return images_path / f"{image_name}.jpg"
    if resolve_mode == "class_subdir":
        return images_path / class_dir / image_name
    if resolve_mode == "mixed_class_or_prefixed":
        if "/" in image_name:
            return images_path / image_name
        return images_path / class_dir / image_name
    if resolve_mode == "prefixed_or_flat":
        return images_path / image_name

    raise ValueError(f"Unsupported resolve_mode: {resolve_mode}")


def _validate_paths(paths: list[Path], split_name: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(f"Missing {split_name} image(s):\n{examples}")
