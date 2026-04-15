from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from transcription import format_timestamp


class VideoRAG:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        index_root: str = "cache/indexes",
        max_segment_candidates: int = 3,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.max_segment_candidates = max_segment_candidates
        self.index_root = Path(index_root)
        self.index_root.mkdir(parents=True, exist_ok=True)

        embedding_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": True, "batch_size": 32}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=embedding_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=260,
            chunk_overlap=40,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.generator = self._build_generator()

    def estimate_timestamps(
        self,
        query: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        duration = int(metadata.get("duration") or 0)
        if duration <= 0:
            raise ValueError("Video duration is unavailable, so timestamps cannot be estimated.")

        title = metadata.get("title", "")
        description = (metadata.get("description", "") or "")[:4000]
        prompt = f"""You estimate likely video time ranges for retrieval.
Return strict JSON with this schema:
{{
  "candidates": [
    {{"start_time": 0, "end_time": 60, "confidence": 0.0, "reason": "short reason"}}
  ]
}}

Rules:
- Use only the title, description, query, and duration.
- Output 1 to {self.max_segment_candidates} candidates.
- Keep ranges short for fast retrieval, ideally 45 to 180 seconds.
- start_time and end_time are integer seconds.
- If uncertain, return multiple candidate ranges.
- Never exceed duration {duration}.

Query: {query}
Title: {title}
Description:
{description}
"""
        started_at = time.perf_counter()
        result = self.generator(
            prompt,
            max_new_tokens=220,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.02,
            return_full_text=False,
        )
        raw_text = result[0]["generated_text"].strip()
        candidates = self._parse_timestamp_candidates(raw_text, duration)
        if not candidates:
            candidates = self._heuristic_timestamp_candidates(query, metadata)

        logger.info(
            "Timestamp estimation completed in {:.2f}s with {} candidate(s)",
            time.perf_counter() - started_at,
            len(candidates),
        )
        return {"candidates": candidates}

    def build_or_load_index(
        self,
        transcript: dict[str, Any],
        video_id: str,
        segment_key: str,
        force_rebuild: bool = False,
    ) -> FAISS:
        embedding_key = self._safe_name(self.embedding_model_name)
        index_dir = self.index_root / video_id / segment_key / embedding_key
        if index_dir.exists() and not force_rebuild:
            logger.info("Loading FAISS index from {}", index_dir)
            return FAISS.load_local(
                str(index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        started_at = time.perf_counter()
        logger.info("Building new FAISS index for segment_key={}", segment_key)
        documents = self._build_documents(transcript)
        if not documents:
            raise ValueError("No transcript segments available to index.")

        vectorstore = self._build_vectorstore(documents)
        index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_dir))

        metadata_path = index_dir / "index_metadata.json"
        metadata = {
            "video_id": video_id,
            "segment_key": segment_key,
            "embedding_model_name": self.embedding_model_name,
            "llm_model_name": self.llm_model_name,
            "document_count": len(documents),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(
            "Embedding and index build completed in {:.2f}s for {}",
            time.perf_counter() - started_at,
            segment_key,
        )
        return vectorstore

    def answer_question(
        self,
        vectorstore: FAISS,
        question: str,
        top_k: int = 4,
    ) -> dict[str, Any]:
        matches = vectorstore.similarity_search_with_score(question, k=top_k)
        if not matches:
            return self._not_found_response()

        relevant_docs = [doc for doc, _ in matches]
        context = self._build_context(relevant_docs)
        if not context.strip():
            return self._not_found_response()

        prompt = self._build_prompt(question=question, context=context)
        logger.info("Generating answer with {} retrieved chunks", len(relevant_docs))
        result = self.generator(
            prompt,
            max_new_tokens=180,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        raw_text = result[0]["generated_text"].strip()
        answer_payload = self._parse_generation(raw_text, relevant_docs)
        answer_payload["retrieved_chunks"] = [
            {
                "text": doc.page_content,
                "timestamp": doc.metadata.get("start_ts"),
                "score_hint": score,
            }
            for doc, score in matches
        ]
        return answer_payload

    def _build_documents(self, transcript: dict[str, Any]) -> list[Document]:
        documents: list[Document] = []
        for idx, segment in enumerate(transcript.get("segments", [])):
            text = segment.get("text", "").strip()
            if not text:
                continue
            metadata = {
                "segment_index": idx,
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
                "start_ts": segment.get("start_ts", format_timestamp(segment.get("start"))),
                "end_ts": segment.get("end_ts", format_timestamp(segment.get("end"))),
            }
            base_doc = Document(page_content=text, metadata=metadata)
            split_docs = self.text_splitter.split_documents([base_doc])
            documents.extend(split_docs)
        return documents

    def _build_vectorstore(self, documents: list[Document]) -> FAISS:
        started_at = time.perf_counter()
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        unique_texts = list(dict.fromkeys(texts))
        embedded_unique = self._embed_texts_in_batches(unique_texts)
        vector_pairs = [(text, embedded_unique[text]) for text in texts]
        logger.info(
            "Embedded {} chunks ({} unique) in {:.2f}s",
            len(texts),
            len(unique_texts),
            time.perf_counter() - started_at,
        )
        return FAISS.from_embeddings(
            text_embeddings=vector_pairs,
            embedding=self.embeddings,
            metadatas=metadatas,
        )

    def _embed_texts_in_batches(self, texts: list[str]) -> dict[str, list[float]]:
        if not texts:
            return {}

        batch_size = 32
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        max_workers = 1 if torch.cuda.is_available() else min(4, len(batches))

        def embed_batch(batch: list[str]) -> dict[str, list[float]]:
            vectors = self.embeddings.embed_documents(batch)
            return dict(zip(batch, vectors, strict=False))

        embeddings_by_text: dict[str, list[float]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_result in executor.map(embed_batch, batches):
                embeddings_by_text.update(batch_result)
        return embeddings_by_text

    def _build_generator(self):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            try:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["device_map"] = "auto"
                logger.info("Loading 4-bit quantized LLM on GPU")
            except Exception:
                logger.warning("Falling back to float16 GPU loading for LLM")
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                **model_kwargs,
            )
        except Exception:
            if torch.cuda.is_available():
                logger.warning("Quantized LLM load failed, retrying without quantization")
                fallback_kwargs = {
                    "low_cpu_mem_usage": True,
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                }
            else:
                raise
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                **fallback_kwargs,
            )
        return pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    def _parse_timestamp_candidates(
        self,
        generated_text: str,
        duration: int,
    ) -> list[dict[str, Any]]:
        match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if not match:
            return []

        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        raw_candidates = payload.get("candidates", [])
        candidates: list[dict[str, Any]] = []
        for candidate in raw_candidates[: self.max_segment_candidates]:
            start = max(0, int(candidate.get("start_time", 0)))
            end = min(duration, int(candidate.get("end_time", start + 60)))
            if end <= start:
                continue
            candidates.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "confidence": float(candidate.get("confidence", 0.0) or 0.0),
                    "reason": str(candidate.get("reason", "")).strip(),
                }
            )
        return candidates

    def _heuristic_timestamp_candidates(
        self,
        query: str,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        duration = int(metadata.get("duration") or 0)
        description = metadata.get("description", "") or ""
        query_terms = {
            term.lower()
            for term in re.findall(r"[a-zA-Z0-9]+", query)
            if len(term) > 2
        }
        candidates: list[dict[str, Any]] = []

        for line in description.splitlines():
            stamps = re.findall(r"(\d{1,2}:\d{2}(?::\d{2})?)", line)
            if not stamps:
                continue
            line_terms = {
                term.lower()
                for term in re.findall(r"[a-zA-Z0-9]+", line)
                if len(term) > 2
            }
            if query_terms and query_terms.isdisjoint(line_terms):
                continue
            stamp_seconds = self._timestamp_to_seconds(stamps[0])
            candidates.append(
                {
                    "start_time": max(0, stamp_seconds - 20),
                    "end_time": min(duration, stamp_seconds + 120),
                    "confidence": 0.35,
                    "reason": "Matched a timestamped description line.",
                }
            )

        if candidates:
            return candidates[: self.max_segment_candidates]

        window = min(180, max(60, duration // 6 if duration else 90))
        default_points = [0, max(0, duration // 3), max(0, (2 * duration) // 3)]
        for start in default_points[: self.max_segment_candidates]:
            end = min(duration, start + window)
            if end > start:
                candidates.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "confidence": 0.1,
                        "reason": "Fallback range because metadata did not strongly localize the query.",
                    }
                )
        return candidates

    @staticmethod
    def _build_context(documents: list[Document]) -> str:
        parts: list[str] = []
        for doc in documents:
            start_ts = doc.metadata.get("start_ts", "00:00:00")
            end_ts = doc.metadata.get("end_ts", start_ts)
            parts.append(f"[{start_ts} - {end_ts}] {doc.page_content.strip()}")
        return "\n".join(parts)

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return f"""You are a strict retrieval QA system.
Answer only from the provided video transcript context.
If the answer is not explicitly present in the context, output exactly:
Answer: Not found in video
Timestamp: [00:00:00](00:00:00)

Rules:
- Do not use external knowledge.
- Keep the answer concise and factual.
- Include exactly one timestamp from the context that best supports the answer.
- Use this exact format:
Answer: <text>
Timestamp: [HH:MM:SS](HH:MM:SS)

Question:
{question}

Context:
{context}
"""

    def _parse_generation(
        self,
        generated_text: str,
        retrieved_docs: list[Document],
    ) -> dict[str, Any]:
        answer_match = re.search(r"Answer:\s*(.+)", generated_text)
        timestamp_match = re.search(
            r"Timestamp:\s*\[(\d{2}:\d{2}:\d{2})\]\((\d{2}:\d{2}:\d{2})\)",
            generated_text,
        )

        if not answer_match or not timestamp_match:
            logger.warning("LLM output format invalid, returning safe fallback")
            return self._not_found_response()

        answer = answer_match.group(1).strip()
        timestamp = timestamp_match.group(1)
        allowed_timestamps = {doc.metadata.get("start_ts") for doc in retrieved_docs}

        if answer.lower() == "not found in video":
            return self._not_found_response()
        if timestamp not in allowed_timestamps:
            logger.warning("LLM timestamp not aligned with retrieved context")
            return self._not_found_response()

        return {
            "answer": answer,
            "timestamp": timestamp,
            "formatted": f"Answer: {answer}\nTimestamp: [{timestamp}]({timestamp})",
        }

    @staticmethod
    def _timestamp_to_seconds(value: str) -> int:
        parts = [int(part) for part in value.split(":")]
        if len(parts) == 2:
            minutes, seconds = parts
            return (minutes * 60) + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return (hours * 3600) + (minutes * 60) + seconds
        return 0

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()

    @staticmethod
    def _not_found_response() -> dict[str, Any]:
        timestamp = "00:00:00"
        return {
            "answer": "Not found in video",
            "timestamp": timestamp,
            "formatted": f"Answer: Not found in video\nTimestamp: [{timestamp}]({timestamp})",
        }
