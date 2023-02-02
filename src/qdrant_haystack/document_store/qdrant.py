import logging
import uuid
from typing import Optional, Dict, List, Union, Generator, Any

import numpy as np
import qdrant_client
from haystack import Document, Label
from haystack.document_stores.base import get_batches_from_generator
from haystack.errors import DocumentStoreError
from haystack.schema import FilterType

from qdrant_client.http import models as rest
from haystack.document_stores import BaseDocumentStore
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QdrantStoreError(DocumentStoreError):
    pass


class QdrantDocumentStore(BaseDocumentStore):
    SIMILARITY = {
        "cosine": rest.Distance.COSINE,
        "dot_product": rest.Distance.DOT,
        "l2": rest.Distance.EUCLID,
    }
    UUID_NAMESPACE = uuid.UUID("3896d314-1e95-4a3a-b45a-945f9f0b541d")

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        collection_name: str = "Document",
        embedding_dim: int = 768,
        content_field: str = "content",
        name_field: str = "name",
        embedding_field: str = "vector",
        similarity: str = "cosine",
        return_embedding: bool = False,
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.client = qdrant_client.QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            **kwargs,
        )

        self._set_up_collection(
            collection_name, embedding_dim, recreate_index, similarity
        )

        self.embedding_dim = embedding_dim
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.similarity = similarity
        self.index = collection_name
        self.return_embedding = return_embedding
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        return list(
            self.get_all_documents_generator(
                index, filters, return_embedding, batch_size, headers
            )
        )

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        raise NotImplementedError

    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        raise NotImplementedError

    def get_document_by_id(
        self,
        id: str,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Document]:
        raise NotImplementedError

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        raise NotImplementedError

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        raise NotImplementedError

    def get_label_count(
        self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> int:
        raise NotImplementedError

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        raise NotImplementedError

    def update_document_meta(
        self, id: str, meta: Dict[str, Any], index: Optional[str] = None
    ):
        raise NotImplementedError

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        index = index or self.index
        self._set_up_collection(index, self.embedding_dim, False, self.similarity)
        field_map = self._create_document_field_map()

        duplicate_documents = duplicate_documents or self.duplicate_documents
        assert (
            duplicate_documents in self.duplicate_documents_options
        ), f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}"

        if len(documents) == 0:
            logger.warning(
                "Calling QdrantDocumentStore.write_documents() with empty list"
            )
            return

        document_objects = [
            Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d
            for d in documents
        ]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects,
            index=index,
            duplicate_documents=duplicate_documents,
        )
        # TODO: make sure all the documents have the embeddings, otherwise create fake ones
        # TODO: convert Document instances to Qdrant batches

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        with tqdm(
            total=len(document_objects), disable=not self.progress_bar
        ) as progress_bar:
            for document_batch in batched_documents:
                payloads = [doc.to_dict(field_map=field_map) for doc in document_batch]
                vectors = [
                    (
                        payload.pop(self.embedding_field)
                        or np.random.random(self.embedding_dim)
                    ).tolist()
                    for payload in payloads
                ]
                # TODO: move the conversion to a separate package
                ids = [
                    uuid.uuid5(self.UUID_NAMESPACE, payload.pop("id")).hex
                    for payload in payloads
                ]

                # TODO: handle duplicate_documents differently
                response = self.client.upsert(
                    collection_name=index,
                    points=rest.Batch(
                        ids=ids,
                        vectors=vectors,
                        payloads=payloads,
                    ),
                )

                # TODO: handle errors in response

                progress_bar.update(batch_size)
        progress_bar.close()

    def delete_index(self, index: str):
        raise NotImplementedError

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().delete_all_documents(index, filters, headers)

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError

    def _create_document_field_map(self) -> Dict:
        return {
            self.name_field: "name",
            self.content_field: "content",
            self.embedding_field: "embedding",
        }

    def _get_distance(self, similarity: str) -> rest.Distance:
        try:
            return self.SIMILARITY[similarity]
        except KeyError:
            raise QdrantStoreError(
                f"Provided similarity '{similarity}' is not supported by Qdrant "
                f"document store. Please choose one of the options: "
                f"{', '.join(self.SIMILARITY.keys())}"
            )

    def _set_up_collection(
        self,
        collection_name: str,
        embedding_dim: int,
        recreate_collection: bool,
        similarity: str,
    ):
        distance = self._get_distance(similarity)

        if recreate_collection:
            # There is no need to verify the current configuration of that
            # collection. It might be just recreated again.
            self._recreate_collection(collection_name, distance, embedding_dim)
            return

        try:
            # Check if the collection already exists and validate its
            # current configuration with the parameters.
            collection_info = self.client.get_collection(collection_name)
            current_distance = collection_info.config.params.vectors.distance
            current_vector_size = collection_info.config.params.vectors.size

            if current_distance != distance:
                raise ValueError(
                    f"Collection '{collection_name}' already exists in Qdrant, "
                    f"but it is configured with a similarity '{current_distance.name}'. "
                    f"If you want to use that collection, but with a different "
                    f"similarity, please set `recreate_collection=True` argument."
                )

            if current_vector_size != embedding_dim:
                raise ValueError(
                    f"Collection '{collection_name}' already exists in Qdrant, "
                    f"but it is configured with a vector size '{current_vector_size}'. "
                    f"If you want to use that collection, but with a different "
                    f"vector size, please set `recreate_collection=True` argument."
                )
        except UnexpectedResponse:
            # That indicates the collection does not exist, so it can be
            # safely created with any configuration.
            self._recreate_collection(collection_name, distance, embedding_dim)

    def _recreate_collection(self, collection_name, distance, embedding_dim):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=embedding_dim,
                distance=distance,
            ),
        )
