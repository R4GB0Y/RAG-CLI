from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    ocr_model: str
    embedding_model: str
    chat_model: str
    similarity_threshold: float
    chunk_size: int
    chunk_overlap: int
    text_layer_min_chars: int
    force_ocr: bool
    data_dir: str
    chroma_dir: str
    text_cache_dir: str


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes")


def load_config() -> Config:
    openai_api_key = _require("OPENAI_API_KEY")

    similarity_threshold = _get_float("SIMILARITY_THRESHOLD", 0.5)
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            f"SIMILARITY_THRESHOLD must be between 0 and 1, got {similarity_threshold}"
        )

    chunk_size = _get_int("CHUNK_SIZE", 500)
    if chunk_size <= 0:
        raise ValueError(f"CHUNK_SIZE must be a positive integer, got {chunk_size}")

    chunk_overlap = _get_int("CHUNK_OVERLAP", 50)
    if chunk_overlap < 0:
        raise ValueError(f"CHUNK_OVERLAP must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE ({chunk_size})"
        )

    text_layer_min_chars = _get_int("TEXT_LAYER_MIN_CHARS", 100)
    if text_layer_min_chars <= 0:
        raise ValueError(
            f"TEXT_LAYER_MIN_CHARS must be a positive integer, got {text_layer_min_chars}"
        )

    return Config(
        openai_api_key=openai_api_key,
        ocr_model=_get_str("OCR_MODEL", "gpt-4o-mini"),
        embedding_model=_get_str("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=_get_str("CHAT_MODEL", "gpt-4o-mini"),
        similarity_threshold=similarity_threshold,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        text_layer_min_chars=text_layer_min_chars,
        force_ocr=_get_bool("FORCE_OCR", False),
        data_dir=_get_str("DATA_DIR", "data"),
        chroma_dir=_get_str("CHROMA_DIR", ".chroma"),
        text_cache_dir=_get_str("TEXT_CACHE_DIR", ".text_cache"),
    )
