from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from loguru import logger

from audio import AudioDownloader
from rag import VideoRAG
from transcription import FasterWhisperTranscriber


APP_TITLE = "YouTube RAG with Hugging Face"
CACHE_DIR = Path("cache")
LOG_DIR = CACHE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
if not getattr(logger, "_yt_rag_logger_configured", False):
    logger.add(LOG_DIR / "app.log", rotation="5 MB", retention=5)
    logger._yt_rag_logger_configured = True


@st.cache_resource(show_spinner=False)
def get_audio_downloader() -> AudioDownloader:
    return AudioDownloader()


@st.cache_resource(show_spinner=True)
def get_transcriber(model_name: str) -> FasterWhisperTranscriber:
    return FasterWhisperTranscriber(model_name=model_name)


@st.cache_resource(show_spinner=True)
def get_rag_engine(embedding_model: str, llm_model: str) -> VideoRAG:
    return VideoRAG(
        embedding_model_name=embedding_model,
        llm_model_name=llm_model,
    )


def initialize_state() -> None:
    st.session_state.setdefault("metadata", None)
    st.session_state.setdefault("artifact", None)
    st.session_state.setdefault("transcript", None)
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("last_candidates", [])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎥", layout="wide")
    initialize_state()

    whisper_model = os.getenv("YT_RAG_WHISPER_MODEL", "small")
    embedding_model = os.getenv(
        "YT_RAG_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    llm_model = os.getenv("YT_RAG_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    st.title(APP_TITLE)
    st.caption(
        "Local YouTube RAG using yt-dlp metadata, partial FFmpeg segment downloads, "
        "faster-whisper, FAISS, and a local Hugging Face LLM."
    )

    with st.sidebar:
        st.subheader("Configuration")
        st.code(
            "\n".join(
                [
                    f"faster-whisper: {whisper_model}",
                    f"Embeddings: {embedding_model}",
                    f"LLM: {llm_model}",
                ]
            ),
            language="text",
        )
        st.info(
            "Each question runs segment-based retrieval: metadata -> timestamp estimate -> "
            "segment download -> transcription -> retrieval -> answer."
        )

    downloader = get_audio_downloader()
    rag_engine = get_rag_engine(embedding_model, llm_model)

    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    rebuild_index = st.checkbox("Force rebuild cached segment index", value=False)
    top_k = st.slider("Top-k retrieved chunks", min_value=2, max_value=8, value=4)

    if st.button("Load Video Metadata", type="primary", use_container_width=True):
        if not url.strip():
            st.error("Please enter a valid YouTube URL.")
        else:
            try:
                with st.spinner("Extracting metadata..."):
                    metadata = downloader.get_video_metadata(url.strip())
                    st.session_state["metadata"] = metadata
                    st.session_state["artifact"] = None
                    st.session_state["transcript"] = None
                    st.session_state["last_response"] = None
                    st.session_state["last_candidates"] = []
            except Exception as exc:
                logger.exception("Metadata extraction failed")
                st.error(str(exc))

    metadata = st.session_state.get("metadata")
    if metadata:
        st.subheader("Video")
        st.write(f"Title: {metadata.get('title')}")
        st.write(f"Video ID: `{metadata.get('id')}`")
        st.write(f"Duration: `{metadata.get('duration', 0)} seconds`")
        if metadata.get("description"):
            st.text_area(
                "Description",
                value=metadata["description"][:1500],
                height=180,
                disabled=True,
            )

    question = st.text_input("Ask a question about the video")

    if st.button("Ask", use_container_width=True):
        if not url.strip():
            st.error("Please enter a valid YouTube URL.")
        elif not question.strip():
            st.error("Enter a question first.")
        else:
            try:
                total_started_at = time.perf_counter()
                with st.status("Running smart segment retrieval...", expanded=True) as status:
                    current_metadata = metadata
                    if not current_metadata:
                        status.write("Extracting video metadata")
                        current_metadata = downloader.get_video_metadata(url.strip())
                        st.session_state["metadata"] = current_metadata

                    status.write("Estimating relevant timestamps from metadata")
                    estimated = rag_engine.estimate_timestamps(question.strip(), current_metadata)
                    candidates = estimated.get("candidates", [])
                    st.session_state["last_candidates"] = candidates
                    if not candidates:
                        raise RuntimeError("Could not estimate any candidate timestamp ranges.")

                    response = None
                    selected_artifact = None
                    selected_transcript = None

                    for candidate in candidates:
                        start_time = int(candidate["start_time"])
                        end_time = int(candidate["end_time"])
                        status.write(
                            f"Downloading segment {start_time}s -> {end_time}s"
                        )
                        artifact = downloader.download_segment(
                            youtube_url=url.strip(),
                            start_time=start_time,
                            end_time=end_time,
                            metadata=current_metadata,
                        )

                        status.write("Transcribing selected segment")
                        transcriber = get_transcriber(whisper_model)
                        transcript = transcriber.transcribe(
                            audio_path=artifact.audio_path,
                            cache_key=artifact.cache_key,
                            segment_start=artifact.segment_start,
                        )

                        status.write("Building or loading cached embeddings")
                        vectorstore = rag_engine.build_or_load_index(
                            transcript=transcript,
                            video_id=artifact.video_id,
                            segment_key=artifact.cache_key,
                            force_rebuild=rebuild_index,
                        )

                        status.write("Retrieving answer from segment context")
                        candidate_response = rag_engine.answer_question(
                            vectorstore=vectorstore,
                            question=question.strip(),
                            top_k=top_k,
                        )
                        if candidate_response["answer"] != "Not found in video":
                            response = candidate_response
                            selected_artifact = artifact
                            selected_transcript = transcript
                            break

                        if response is None:
                            response = candidate_response
                            selected_artifact = artifact
                            selected_transcript = transcript

                    status.update(label="Query completed", state="complete")

                total_elapsed = time.perf_counter() - total_started_at
                logger.info("Total response time: {:.2f}s", total_elapsed)

                st.session_state["artifact"] = selected_artifact
                st.session_state["transcript"] = selected_transcript
                st.session_state["last_response"] = response
            except ValueError as exc:
                logger.exception("Question answering failed with validation error")
                st.error(str(exc))
            except Exception as exc:
                logger.exception("Question answering failed")
                st.error(str(exc))

    artifact = st.session_state.get("artifact")
    transcript = st.session_state.get("transcript")
    response = st.session_state.get("last_response")

    if artifact:
        st.subheader("Last Processed Segment")
        st.write(
            f"Range: `{artifact.segment_start}`s -> `{artifact.segment_end}`s | "
            f"Cache key: `{artifact.cache_key}`"
        )

    if transcript:
        st.subheader("Segment Transcript Preview")
        st.write(f"Detected language: `{transcript.get('language') or 'auto/unknown'}`")
        st.text_area(
            "Transcript",
            value=transcript.get("text", ""),
            height=180,
            disabled=True,
        )

    if response:
        st.subheader("Answer")
        st.code(response["formatted"], language="text")

        with st.expander("Retrieved Context"):
            for item in response.get("retrieved_chunks", []):
                st.markdown(f"**{item['timestamp']}**\n\n{item['text']}")

    candidates = st.session_state.get("last_candidates", [])
    if candidates:
        with st.expander("Estimated Timestamp Candidates"):
            for candidate in candidates:
                st.write(
                    f"{candidate['start_time']}s -> {candidate['end_time']}s "
                    f"(confidence={candidate.get('confidence', 0.0):.2f}) "
                    f"{candidate.get('reason', '')}"
                )


if __name__ == "__main__":
    main()
