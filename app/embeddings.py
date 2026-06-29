from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


LOCAL_BGE_M3_PATH = Path("models/bge-m3")


def _default_embedding_backend() -> str:
    return "sentence-transformers" if LOCAL_BGE_M3_PATH.exists() else "hashing"


def _default_embedding_model() -> str:
    return str(LOCAL_BGE_M3_PATH) if LOCAL_BGE_M3_PATH.exists() else "BAAI/bge-m3"


EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", _default_embedding_backend())
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _default_embedding_model())
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:1234/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LOCAL_API_KEY", "lm-studio"))
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "60"))

_sentence_model_cache = {}


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = EMBEDDING_BACKEND
    model: str = EMBEDDING_MODEL
    batch_size: int = EMBEDDING_BATCH_SIZE
    dim: int = EMBEDDING_DIM
    base_url: str = EMBEDDING_BASE_URL
    api_key: str = EMBEDDING_API_KEY
    timeout: float = EMBEDDING_TIMEOUT
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    local_only: bool = os.getenv("EMBEDDING_LOCAL_ONLY", "1") == "1"


def default_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9+\-.%]*", text.lower())


def _hashing_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _get_sentence_model(config: EmbeddingConfig):
    cache_key = (config.model, config.device, config.local_only)
    if cache_key not in _sentence_model_cache:
        from sentence_transformers import SentenceTransformer

        _sentence_model_cache[cache_key] = SentenceTransformer(
            config.model,
            local_files_only=config.local_only,
            device=config.device,
        )
    return _sentence_model_cache[cache_key]


def get_embedding_settings(config: EmbeddingConfig | None = None) -> dict[str, str]:
    config = config or default_embedding_config()
    return {
        "backend": config.backend,
        "model": config.model,
        "device": config.device,
        "local_only": "1" if config.local_only else "0",
        "hashing_dim": str(config.dim),
        "batch_size": str(config.batch_size),
        "base_url": config.base_url,
        "timeout": str(config.timeout),
    }


def _external_embeddings(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    base_url = config.base_url.rstrip("/")
    response = requests.post(
        f"{base_url}/embeddings",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "input": texts,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    payload = response.json()
    data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in data]


def get_embeddings(
    texts: list[str],
    config: EmbeddingConfig | None = None,
) -> list[list[float]]:
    """Формирует embeddings пачкой, чтобы не вызывать модель отдельно на каждый чанк."""
    config = config or default_embedding_config()
    if config.backend == "sentence-transformers":
        embeddings = _get_sentence_model(config).encode(
            texts,
            convert_to_numpy=True,
            batch_size=config.batch_size,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    if config.backend == "openai-compatible":
        return _external_embeddings(texts, config)
    return [_hashing_embedding(text, dim=config.dim) for text in texts]


def get_embedding(
    text: str,
    config: EmbeddingConfig | None = None,
) -> list[float]:
    """Формирует embedding без сетевой зависимости по умолчанию."""
    return get_embeddings([text], config=config)[0]
