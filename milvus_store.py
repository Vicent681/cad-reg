#!/usr/bin/env python3
"""Milvus helper utilities for storing and retrieving CAD connectivity block embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

DEFAULT_COLLECTION = os.getenv("CAD_RAG_COLLECTION", "cad_connectivity_blocks")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")


def _connect() -> None:
    if connections.has_connection("default"):
        return
    connections.connect(
        alias="default",
        uri=MILVUS_URI,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
    )


def _create_collection(collection_name: str, dim: int) -> Collection:
    fields = [
        FieldSchema(
            name="block_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=128,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
        FieldSchema(
            name="doc_id",
            dtype=DataType.VARCHAR,
            max_length=128,
        ),
        FieldSchema(
            name="page_number",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="bbox",
            dtype=DataType.VARCHAR,
            max_length=128,
        ),
        FieldSchema(
            name="image_path",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="summary",
            dtype=DataType.VARCHAR,
            max_length=2048,
        ),
    ]
    schema = CollectionSchema(fields, description="CAD connectivity block embeddings")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024},
        },
    )
    collection.load()
    return collection


def _ensure_collection(collection_name: str, dim: int) -> Collection:
    _connect()
    if not utility.has_collection(collection_name):
        return _create_collection(collection_name, dim)

    collection = Collection(collection_name)
    embedding_field = next(
        (field for field in collection.schema.fields if field.name == "embedding"),
        None,
    )
    existing_dim = None
    if embedding_field is not None:
        params = getattr(embedding_field, "params", None) or {}
        existing_dim = params.get("dim") or params.get("dimension")
        if existing_dim is not None:
            try:
                existing_dim = int(existing_dim)
            except (TypeError, ValueError):
                existing_dim = None

    if existing_dim is not None and existing_dim != dim:
        # Drop and recreate collection to match the new embedding dimension.
        collection.release()
        utility.drop_collection(collection_name)
        return _create_collection(collection_name, dim)

    collection.load()
    return collection


def insert_blocks(records: Sequence[dict], embeddings, collection_name: str = DEFAULT_COLLECTION) -> int:
    if not records:
        return 0
    dim = len(embeddings[0])
    collection = _ensure_collection(collection_name, dim)
    block_ids = [record["block_id"] for record in records]
    doc_ids = [record["doc_id"] for record in records]
    page_numbers = [int(record.get("page_number", 0)) for record in records]
    bbox_values = [",".join(str(v) for v in record["bbox"]) for record in records]
    image_paths = [record["image_path"] for record in records]
    summaries = [record["summary"] for record in records]

    collection.insert(
        [
            block_ids,
            embeddings,
            doc_ids,
            page_numbers,
            bbox_values,
            image_paths,
            summaries,
        ]
    )
    collection.flush()
    return len(records)


def search_blocks(
    query_embedding,
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    score_threshold: float | None = None,
    doc_ids: Sequence[str] | None = None,
) -> List[dict]:
    _connect()
    if not utility.has_collection(collection_name):
        return []
    collection = Collection(collection_name)
    if not collection.has_index():
        # ensure index exists; fallback to auto creation
        dim = len(query_embedding)
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 1024},
            },
        )
    collection.load()
    search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
    expr = None
    if doc_ids:
        quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
        expr = f"doc_id in [{quoted}]"

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["doc_id", "page_number", "bbox", "image_path", "summary"],
    )
    hits = results[0] if results else []
    parsed: List[dict] = []
    for hit in hits:
        score = float(hit.score)
        if score_threshold is not None and score < score_threshold:
            continue
        parsed.append(
            {
                "block_id": hit.id,
                "score": score,
                "doc_id": hit.entity.get("doc_id"),
                "page_number": int(hit.entity.get("page_number", 0)),
                "bbox": hit.entity.get("bbox"),
                "image_path": hit.entity.get("image_path"),
                "summary": hit.entity.get("summary"),
            }
        )
    return parsed
