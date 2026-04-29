"""Optional source-video and extracted-frame window datasets for SurgWMBench."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from planet_torch_surgwmbench.data.surgwmbench import (
    _frame_local_index,
    _frame_path_value,
    load_json,
    read_jsonl_manifest,
    resolve_dataset_path,
)
from planet_torch_surgwmbench.data.transforms import pil_to_float_tensor, target_size_hw


def _read_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl_manifest(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalise_source_video_records(dataset_root: Path, payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("videos"), list):
            rows = payload["videos"]
        elif isinstance(payload.get("source_videos"), list):
            rows = payload["source_videos"]
        else:
            rows = list(payload.values()) if all(isinstance(value, dict) for value in payload.values()) else [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("source video manifest must be a JSON object, list, or JSONL rows")

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"source video manifest row {index} is not an object")
        source_video_id = str(row.get("source_video_id", row.get("video_id", row.get("id", f"video_{index:04d}"))))
        raw_path = (
            row.get("source_video_path")
            or row.get("video_left_path")
            or row.get("video_path")
            or row.get("path")
            or f"videos/{source_video_id}/video_left.avi"
        )
        if source_video_id in seen:
            continue
        seen.add(source_video_id)
        path = resolve_dataset_path(dataset_root, raw_path)
        if path is None:
            raise ValueError(f"Missing source video path for {source_video_id}")
        records.append(
            {
                "source_video_id": source_video_id,
                "source_video_path": str(path),
            }
        )
    return records


def _image_array_to_tensor(array: np.ndarray, image_size: int | tuple[int, int] | None) -> torch.Tensor:
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    size_hw = target_size_hw(image_size)
    if size_hw is not None and (image.height, image.width) != size_hw:
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((size_hw[1], size_hw[0]), int(resampling.BILINEAR))
    return pil_to_float_tensor(image)


class SurgWMBenchRawVideoDataset(Dataset):
    """Load source video windows or fallback windows from extracted clip frames."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        source_video_manifest: str | Path | None = None,
        clip_length: int = 16,
        stride: int = 4,
        image_size: int | tuple[int, int] = 128,
        backend: str = "opencv",
        max_videos: int | None = None,
        max_clips_per_video: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        self.split = split
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        self.image_size = image_size
        self.backend = backend
        self.max_videos = max_videos
        self.max_clips_per_video = max_clips_per_video

        if self.clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if backend not in {"opencv", "frames", "clip_frames", "extracted_frames"}:
            raise ValueError("backend must be one of: opencv, frames, clip_frames, extracted_frames")

        if backend == "opencv":
            self.records = self._build_opencv_index(source_video_manifest)
        else:
            self.records = self._build_frame_fallback_index()
        if not self.records:
            raise ValueError("SurgWMBenchRawVideoDataset built no windows")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record["backend"] == "opencv":
            frames = self._read_opencv_window(record)
        else:
            frames = self._read_frame_window(record)
        return {
            "frames": frames,
            "source_video_id": record["source_video_id"],
            "source_video_path": record["source_video_path"],
            "start_frame": int(record["start_frame"]),
            "frame_indices": torch.as_tensor(record["frame_indices"], dtype=torch.long),
        }

    def _source_records(self, source_video_manifest: str | Path | None) -> list[dict[str, str]]:
        if source_video_manifest is not None:
            manifest_path = Path(source_video_manifest).expanduser()
            manifest_path = manifest_path if manifest_path.is_absolute() else self.dataset_root / manifest_path
            payload = _read_json_or_jsonl(manifest_path)
            records = _normalise_source_video_records(self.dataset_root, payload)
        else:
            metadata_path = self.dataset_root / "metadata" / "source_videos.json"
            if metadata_path.exists():
                records = _normalise_source_video_records(self.dataset_root, load_json(metadata_path))
            else:
                manifest_path = self.dataset_root / "manifests" / f"{self.split}.jsonl"
                entries = read_jsonl_manifest(manifest_path)
                records = _normalise_source_video_records(self.dataset_root, entries)
        if self.max_videos is not None:
            records = records[: max(int(self.max_videos), 0)]
        return records

    def _build_opencv_index(self, source_video_manifest: str | Path | None) -> list[dict[str, Any]]:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for SurgWMBenchRawVideoDataset backend='opencv'. "
                "Install opencv-python or use backend='frames'."
            ) from exc

        records: list[dict[str, Any]] = []
        for source in self._source_records(source_video_manifest):
            video_path = Path(source["source_video_path"])
            if not video_path.exists():
                raise FileNotFoundError(f"Source video not found: {video_path}")
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"Could not open source video with OpenCV: {video_path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if frame_count <= 0:
                raise ValueError(f"OpenCV reported no frames for source video: {video_path}")
            starts = self._window_starts(frame_count)
            if self.max_clips_per_video is not None:
                starts = starts[: max(int(self.max_clips_per_video), 0)]
            for start in starts:
                indices = list(range(start, min(start + self.clip_length, frame_count)))
                records.append(
                    {
                        "backend": "opencv",
                        "source_video_id": source["source_video_id"],
                        "source_video_path": str(video_path),
                        "start_frame": start,
                        "frame_indices": indices,
                    }
                )
        return records

    def _build_frame_fallback_index(self) -> list[dict[str, Any]]:
        manifest_path = self.dataset_root / "manifests" / f"{self.split}.jsonl"
        entries = read_jsonl_manifest(manifest_path)
        source_counts: dict[str, int] = {}
        records: list[dict[str, Any]] = []

        for entry in entries:
            source_video_id = str(entry.get("source_video_id", ""))
            if self.max_videos is not None and source_video_id not in source_counts and len(source_counts) >= int(self.max_videos):
                continue
            if self.max_clips_per_video is not None and source_counts.get(source_video_id, 0) >= int(self.max_clips_per_video):
                continue

            annotation_path = resolve_dataset_path(self.dataset_root, entry.get("annotation_path"))
            if annotation_path is None or not annotation_path.exists():
                raise FileNotFoundError(f"Annotation not found for raw-video frame fallback: {annotation_path}")
            annotation = load_json(annotation_path)
            frames = annotation.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"Annotation has no frames[] for frame fallback: {annotation_path}")

            frame_paths = self._frame_paths_for_entry(entry, frames)
            starts = self._window_starts(len(frame_paths))
            for start in starts:
                if self.max_clips_per_video is not None and source_counts.get(source_video_id, 0) >= int(self.max_clips_per_video):
                    break
                end = min(start + self.clip_length, len(frame_paths))
                selected_paths = frame_paths[start:end]
                selected_frames = frames[start:end]
                frame_indices = [_frame_local_index(frame, start + offset) for offset, frame in enumerate(selected_frames)]
                records.append(
                    {
                        "backend": "frames",
                        "source_video_id": source_video_id,
                        "source_video_path": str(entry.get("source_video_path", annotation.get("source_video_path", ""))),
                        "start_frame": int(frame_indices[0]),
                        "frame_indices": frame_indices,
                        "frame_paths": [str(path) for path in selected_paths],
                    }
                )
                source_counts[source_video_id] = source_counts.get(source_video_id, 0) + 1
        return records

    def _window_starts(self, length: int) -> list[int]:
        if length <= 0:
            return []
        if length <= self.clip_length:
            return [0]
        starts = list(range(0, length - self.clip_length + 1, self.stride))
        if not starts:
            starts = [0]
        return starts

    def _frame_paths_for_entry(self, entry: dict[str, Any], frames: list[Any]) -> list[Path]:
        frames_dir = resolve_dataset_path(self.dataset_root, entry.get("frames_dir"))
        if frames_dir is None:
            raise ValueError("Manifest entry is missing frames_dir")

        paths: list[Path] = []
        for fallback_idx, frame in enumerate(frames):
            local_idx = _frame_local_index(frame, fallback_idx)
            value = _frame_path_value(frame)
            if value is None:
                path = frames_dir / f"{local_idx:06d}.jpg"
            else:
                candidate = Path(value)
                path = candidate if candidate.is_absolute() else self.dataset_root / candidate
                if not path.exists() and not candidate.is_absolute():
                    path = frames_dir / value
            if not path.exists():
                raise FileNotFoundError(f"Frame image not found for frame fallback: {path}")
            paths.append(path)
        return paths

    def _read_opencv_window(self, record: dict[str, Any]) -> torch.Tensor:
        cv2 = importlib.import_module("cv2")
        capture = cv2.VideoCapture(record["source_video_path"])
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open source video with OpenCV: {record['source_video_path']}")

        frames: list[torch.Tensor] = []
        for frame_idx in record["frame_indices"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = capture.read()
            if not ok:
                capture.release()
                raise ValueError(f"Could not read frame {frame_idx} from {record['source_video_path']}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(_image_array_to_tensor(frame_rgb, self.image_size))
        capture.release()
        return torch.stack(frames, dim=0)

    def _read_frame_window(self, record: dict[str, Any]) -> torch.Tensor:
        frames: list[torch.Tensor] = []
        for path in record["frame_paths"]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                size_hw = target_size_hw(self.image_size)
                if size_hw is not None and (image.height, image.width) != size_hw:
                    resampling = getattr(Image, "Resampling", Image)
                    image = image.resize((size_hw[1], size_hw[0]), int(resampling.BILINEAR))
                frames.append(pil_to_float_tensor(image))
        return torch.stack(frames, dim=0)


__all__ = ["SurgWMBenchRawVideoDataset"]
