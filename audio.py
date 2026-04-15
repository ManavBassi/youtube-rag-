from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from yt_dlp import YoutubeDL


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


@dataclass(slots=True)
class SegmentArtifact:
    video_id: str
    title: str
    source_url: str
    audio_path: str
    work_dir: str
    metadata_path: str
    segment_start: int
    segment_end: int
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioDownloader:
    def __init__(self, cache_root: str = "cache/videos") -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def get_video_metadata(self, youtube_url: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        logger.info("Extracting video metadata for {}", youtube_url)
        opts = {
            "extract_flat": False,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
        except Exception as exc:
            logger.exception("Unable to fetch video metadata")
            raise RuntimeError("Invalid YouTube URL or metadata extraction failed.") from exc

        if not info or "id" not in info:
            raise ValueError("Invalid YouTube URL. Could not determine video metadata.")

        metadata = {
            "id": info["id"],
            "title": info.get("title", info["id"]),
            "description": info.get("description", "") or "",
            "duration": int(info.get("duration") or 0),
            "webpage_url": info.get("webpage_url", youtube_url),
        }

        work_dir = self.cache_root / metadata["id"]
        work_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = work_dir / "video_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(
            "Metadata extraction completed in {:.2f}s for video_id={}",
            time.perf_counter() - started_at,
            metadata["id"],
        )
        return metadata

    def download_segment(
        self,
        youtube_url: str,
        start_time: int,
        end_time: int,
        metadata: dict[str, Any] | None = None,
    ) -> SegmentArtifact:
        if start_time < 0 or end_time <= start_time:
            raise ValueError("Invalid segment range requested.")

        started_at = time.perf_counter()
        metadata = metadata or self.get_video_metadata(youtube_url)
        video_id = metadata["id"]
        duration = int(metadata.get("duration") or 0)
        safe_end = min(end_time, duration) if duration else end_time
        safe_start = min(start_time, max(0, safe_end - 1))
        segment_duration = safe_end - safe_start
        if segment_duration <= 0:
            raise ValueError("Predicted timestamp range is outside the video duration.")

        cache_key = f"{video_id}_{safe_start:06d}_{safe_end:06d}"
        work_dir = self.cache_root / video_id / "segments" / cache_key
        work_dir.mkdir(parents=True, exist_ok=True)
        audio_path = work_dir / "segment.wav"
        metadata_path = work_dir / "segment_metadata.json"

        artifact = SegmentArtifact(
            video_id=video_id,
            title=metadata.get("title", video_id),
            source_url=youtube_url,
            audio_path=str(audio_path),
            work_dir=str(work_dir),
            metadata_path=str(metadata_path),
            segment_start=safe_start,
            segment_end=safe_end,
            cache_key=cache_key,
        )

        if audio_path.exists():
            logger.info("Reusing cached segment audio for {}", cache_key)
            self._persist_segment_metadata(metadata_path, metadata, artifact)
            return artifact

        stream_url = self._extract_audio_stream_url(youtube_url)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(safe_start),
            "-t",
            str(segment_duration),
            "-i",
            stream_url,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("FFmpeg is required but was not found on this system.") from exc
        except subprocess.CalledProcessError as exc:
            logger.exception("Segment download failed for {}", cache_key)
            error_output = exc.stderr.strip() or exc.stdout.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"Segment download failed: {error_output}") from exc

        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise RuntimeError("Segment download failed: no audio was created.")

        self._persist_segment_metadata(metadata_path, metadata, artifact)
        logger.info(
            "Segment download completed in {:.2f}s for {}",
            time.perf_counter() - started_at,
            cache_key,
        )
        return artifact

    @staticmethod
    def _extract_audio_stream_url(youtube_url: str) -> str:
        opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
        except Exception as exc:
            logger.exception("Unable to resolve direct audio stream")
            raise RuntimeError("Unable to resolve YouTube audio stream for segment download.") from exc

        stream_url = info.get("url")
        if not stream_url:
            raise RuntimeError("yt-dlp did not return a playable audio stream URL.")
        return stream_url

    @staticmethod
    def _persist_segment_metadata(
        metadata_path: Path,
        metadata: dict[str, Any],
        artifact: SegmentArtifact,
    ) -> None:
        payload = {
            "video": {
                "id": metadata.get("id"),
                "title": metadata.get("title"),
                "description": metadata.get("description"),
                "duration": metadata.get("duration"),
                "webpage_url": metadata.get("webpage_url"),
            },
            "artifact": artifact.to_dict(),
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
