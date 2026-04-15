from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel
from loguru import logger


def format_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "00:00:00"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    start: float
    end: float

    @property
    def start_ts(self) -> str:
        return format_timestamp(self.start)

    @property
    def end_ts(self) -> str:
        return format_timestamp(self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str = "small",
        cache_root: str = "cache/transcripts",
    ) -> None:
        self.model_name = model_name
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading faster-whisper model={} on device={}", self.model_name, device)
        self.model = WhisperModel(
            self.model_name,
            device=device,
            compute_type="int8",
            download_root=str(self.cache_root / "models"),
        )

    def transcribe(
        self,
        audio_path: str,
        cache_key: str,
        segment_start: float = 0.0,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        model_key = self._safe_name(self.model_name)
        transcript_path = self.cache_root / f"{cache_key}_{model_key}.json"
        if transcript_path.exists():
            logger.info("Reusing cached transcript for {}", cache_key)
            return json.loads(transcript_path.read_text(encoding="utf-8"))

        logger.info("Starting faster-whisper transcription for {}", audio_path)
        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=1,
                best_of=1,
                vad_filter=True,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            logger.exception("faster-whisper transcription failed")
            raise RuntimeError("Transcription failed for the selected video segment.") from exc

        extracted_segments = self._extract_segments(segments, segment_start)
        full_text = " ".join(segment.text for segment in extracted_segments).strip()
        if not extracted_segments or not full_text:
            raise ValueError("Empty transcription for the selected video segment.")

        payload = {
            "cache_key": cache_key,
            "model_name": self.model_name,
            "language": getattr(info, "language", None),
            "duration": getattr(info, "duration", None),
            "text": full_text,
            "segments": [segment.to_dict() for segment in extracted_segments],
        }
        transcript_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Transcription completed in {:.2f}s for {}",
            time.perf_counter() - started_at,
            cache_key,
        )
        return payload

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()

    @staticmethod
    def _extract_segments(
        raw_segments: Any,
        segment_start: float,
    ) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        for chunk in raw_segments:
            text = (chunk.text or "").strip()
            if not text:
                continue
            start = float(chunk.start or 0.0) + float(segment_start)
            end = float(chunk.end or chunk.start or 0.0) + float(segment_start)
            segments.append(TranscriptSegment(text=text, start=start, end=end))
        return segments
