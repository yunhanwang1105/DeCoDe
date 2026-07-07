#!/usr/bin/env python3
"""Extract one middle frame per UCF101 video.

The DeCoDe UCF reader expects a flat frame directory:

    UCF_frame/<video_id>.jpg

For example:

    UCF_frame/v_VolleyballSpiking_g07_c05.jpg

Run from the UCF101 data root, or pass explicit paths:

    python extract_ucf_middle_frames.py \
        --videos_dir UCF-101 \
        --output_dir UCF_frame
"""

from __future__ import annotations

import argparse
from pathlib import Path


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mpeg", ".mpg", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos_dir",
        type=Path,
        default=Path("UCF-101"),
        help="Root containing UCF101 videos, usually class subfolders with .avi files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("UCF_frame"),
        help="Output directory for flat <video_id>.jpg middle frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite frames that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable

    videos_dir = args.videos_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        path
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise FileNotFoundError(f"No videos found under: {videos_dir}")

    failures = []
    for video_path in tqdm(video_paths, desc="Extracting UCF101 middle frames"):
        output_path = output_dir / f"{video_path.stem}.jpg"
        if output_path.exists() and not args.overwrite:
            continue

        ok = extract_middle_frame(video_path, output_path)
        if not ok:
            failures.append(str(video_path))

    print(f"Processed videos: {len(video_paths)}")
    print(f"Frames written to: {output_dir}")
    if failures:
        print(f"Failed videos: {len(failures)}")
        for path in failures[:20]:
            print(f"  - {path}")
        raise RuntimeError("Some videos could not be processed.")


def extract_middle_frame(video_path: Path, output_path: Path) -> bool:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return False

        middle_index = frame_count // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, middle_index)
        success, frame = capture.read()
        if not success or frame is None:
            return False

        return bool(cv2.imwrite(str(output_path), frame))
    finally:
        capture.release()


if __name__ == "__main__":
    main()
