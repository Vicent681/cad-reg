#!/usr/bin/env python3
"""Helpers to call ModelScope's embedding endpoint via the OpenAI-compatible SDK."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

import numpy as np
from openai import OpenAI


def _embedding_model() -> str:
    return os.getenv("CAD_RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")


def _embedding_base_url() -> str:
    return os.getenv("CAD_RAG_EMBED_BASE_URL") or os.getenv("MODELSCOPE_BASE_URL") or "https://api-inference.modelscope.cn/v1"


def _embedding_api_key() -> str:
    key = (
        os.getenv("CAD_RAG_EMBED_API_KEY")
        or os.getenv("MODELSCOPE_API_KEY")
        or os.getenv("MODELSCOPE_TOKEN")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        raise RuntimeError("缺少嵌入模型的 API Key，请设置 CAD_RAG_EMBED_API_KEY 或 MODELSCOPE_API_KEY。")
    return key


def _resolve_embedding_model(value: str | None) -> str:
    value = (value or "").strip()
    return value or _embedding_model()


def _resolve_embedding_base_url(value: str | None) -> str:
    value = (value or "").strip()
    return value or _embedding_base_url()


def _resolve_embedding_api_key(value: str | None) -> str:
    value = (value or "").strip()
    return value or _embedding_api_key()


@lru_cache(maxsize=4)
def _client_for(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def embed_texts(
    texts: Sequence[str],
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """Return float32 embeddings for the provided texts."""
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    model_id = _resolve_embedding_model(model)
    base_url_value = _resolve_embedding_base_url(base_url)
    api_key_value = _resolve_embedding_api_key(api_key)
    client = _client_for(base_url_value, api_key_value)
    response = client.embeddings.create(
        model=model_id,
        input=list(texts),
        encoding_format="float",
    )
    vectors = []
    for item in response.data:
        embedding = getattr(item, "embedding", None)
        if embedding is None:
            raise RuntimeError("嵌入接口返回异常，缺少 embedding 字段。")
        vectors.append(np.asarray(embedding, dtype="float32"))
    return np.stack(vectors, axis=0)
