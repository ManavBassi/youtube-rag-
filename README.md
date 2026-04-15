# YouTube RAG with Hugging Face

Local retrieval-augmented question answering over YouTube videos using only open-source tooling.

## Features

- `yt-dlp` metadata extraction plus partial segment download with FFmpeg
- Query-driven timestamp estimation with a local Hugging Face LLM
- `faster-whisper` transcription with timestamped segments
- Smaller timestamp-aware transcript chunking with `RecursiveCharacterTextSplitter`
- Sentence-transformer embeddings
- FAISS vector index with per-segment persistence
- Local Hugging Face causal LLM for context-grounded answers
- Streamlit UI with loading indicators and cache reuse

## Project Files

- `audio.py`: metadata extraction and cached segment downloads
- `transcription.py`: faster-whisper ASR and timestamp formatting
- `rag.py`: timestamp estimation, chunking, embeddings, FAISS, retrieval, and answer generation
- `app.py`: Streamlit interface for segment-based retrieval

## Run on Linux

1. Install system dependencies:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Optionally choose lighter or heavier local models:

```bash
export YT_RAG_WHISPER_MODEL=small
export YT_RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export YT_RAG_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
```

5. Start the app:

```bash
streamlit run app.py
```

6. Open the URL shown by Streamlit in VS Code or your browser.

## Notes

- GPU is used automatically when `torch.cuda.is_available()` is true.
- The app caches segment audio, transcripts, model instances, and FAISS indexes by `video_id + timestamp range`.
- Each query follows: metadata -> timestamp estimate -> segment download -> transcription -> retrieval -> answer.
- If the local LLM cannot ground the answer in retrieved chunks, the app returns `Not found in video`.
# youtube-rag-
