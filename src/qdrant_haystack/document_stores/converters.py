import uuid
from typing import Dict, List, Union

import numpy as np
from haystack import Document
from qdrant_client.http import models as rest


class HaystackToQdrant:
    """A converter from Haystack to Qdrant types."""

    UUID_NAMESPACE = uuid.UUID("3896d314-1e95-4a3a-b45a-945f9f0b541d")

    def documents_to_batch(
        self,
        documents: List[Document],
        *,
        embedding_field: str,
        embedding_dim: int,
        field_map: Dict[str, str],
        fill_missing_embeddings: bool = False
    ) -> rest.Batch:
        payloads = [doc.to_dict(field_map=field_map) for doc in documents]
        vectors = [payload.pop(embedding_field) for payload in payloads]
        if fill_missing_embeddings:
            vectors = [
                vector if vector is not None else np.random.random(embedding_dim)
                for vector in vectors
            ]
        vectors = [vector.tolist() for vector in vectors]
        ids = [self.convert_id(payload.get("id")) for payload in payloads]
        return rest.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )

    def convert_id(self, id: str) -> str:
        """
        Converts any string into a UUID-like format in a deterministic way.

        Qdrant does not accept any string as an id, so an internal id has to be
        generated for each point. This is a deterministic way of doing so.
        """
        return uuid.uuid5(self.UUID_NAMESPACE, id).hex


QdrantPoint = Union[rest.ScoredPoint, rest.Record]


class QdrantToHaystack:
    def __init__(self, content_field: str, name_field: str, embedding_field: str):
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field

    def point_to_document(self, point: QdrantPoint) -> Document:
        payload = {**point.payload}
        return Document(
            content=payload.pop(self.content_field),
            content_type=payload.pop("content_type"),
            id=payload.pop("id"),
            meta=payload.pop("meta"),
            score=point.score if hasattr(point, "score") else None,
            embedding=np.array(point.vector) if point.vector else None,
            id_hash_keys=payload.pop("id_hash_keys"),
        )
