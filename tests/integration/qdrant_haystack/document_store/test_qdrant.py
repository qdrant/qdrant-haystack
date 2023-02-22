import uuid
from typing import List
from unittest.mock import MagicMock

import pytest
import numpy as np

from haystack import Document

from qdrant_haystack import QdrantDocumentStore


EMBEDDING_DIM = 768


@pytest.fixture
def data_store() -> QdrantDocumentStore:
    return QdrantDocumentStore(
        url="http://localhost", recreate_index=True, return_embedding=True
    )


@pytest.fixture
def documents() -> List[Document]:
    documents = []
    for i in range(3):
        documents.append(
            Document(
                id=str(uuid.uuid4()),
                content=f"A Foo Document {i}",
                meta={
                    "name": f"name_{i}",
                    "year": "2020",
                    "month": "01",
                    "numbers": [2.0, 4.0],
                },
                embedding=np.random.rand(EMBEDDING_DIM).astype(np.float32),
            )
        )

        documents.append(
            Document(
                id=str(uuid.uuid4()),
                content=f"A Bar Document {i}",
                meta={
                    "name": f"name_{i}",
                    "year": "2021",
                    "month": "02",
                    "numbers": [-2.0, -4.0],
                },
                embedding=np.random.rand(EMBEDDING_DIM).astype(np.float32),
            )
        )

        documents.append(
            Document(
                id=str(uuid.uuid4()),
                content=f"A Baz Document {i}",
                meta={"name": f"name_{i}", "month": "03"},
                embedding=np.random.rand(EMBEDDING_DIM).astype(np.float32),
            )
        )

    return documents


@pytest.mark.integration
def test_ne_filters(data_store: QdrantDocumentStore, documents: List[Document]):
    data_store.write_documents(documents)

    result = data_store.get_all_documents(filters={"year": {"$ne": "2020"}})
    assert len(result) == 6


@pytest.mark.integration
def test_nin_filters(data_store: QdrantDocumentStore, documents: List[Document]):
    data_store.write_documents(documents)

    result = data_store.get_all_documents(
        filters={"year": {"$nin": ["2020", "2021", "n.a."]}}
    )
    assert len(result) == 3


@pytest.mark.integration
def test_delete_index(data_store: QdrantDocumentStore, documents: List[Document]):
    data_store.write_documents(documents, index="custom_index")
    assert data_store.get_document_count(index="custom_index") == len(documents)
    data_store.delete_index(index="custom_index")
    assert data_store.get_document_count(index="custom_index") == 0


@pytest.mark.integration
def test_query_by_embedding(data_store: QdrantDocumentStore, documents: List[Document]):
    data_store.write_documents(documents)

    docs = data_store.query_by_embedding(
        np.random.rand(EMBEDDING_DIM).astype(np.float32)
    )
    assert len(docs) == 9

    docs = data_store.query_by_embedding(
        np.random.rand(EMBEDDING_DIM).astype(np.float32), top_k=1
    )
    assert len(docs) == 1

    docs = data_store.query_by_embedding(
        np.random.rand(EMBEDDING_DIM).astype(np.float32), filters={"name": ["name_1"]}
    )
    assert len(docs) == 3


@pytest.mark.integration
def test_deleting_by_id_or_by_filters(
    data_store: QdrantDocumentStore, documents: List[Document]
):
    data_store.write_documents(documents)
    data_store.get_all_documents = MagicMock(wraps=data_store.get_all_documents)

    assert data_store.get_document_count() == 9

    # Delete a document by its ID. This should bypass the get_all_documents() call
    data_store.delete_documents(ids=[documents[0].id])
    data_store.get_all_documents.assert_not_called()
    assert data_store.get_document_count() == 8

    data_store.get_all_documents.reset_mock()
    # Delete a document with filters. This should bypass the get_all_documents() call
    data_store.delete_documents(filters={"name": ["name_0"]})
    data_store.get_all_documents.assert_not_called()
    assert data_store.get_document_count() == 6


@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["cosine", "l2", "dot_product"])
def test_similarity_existing_index(similarity: str):
    """Testing non-matching similarity"""
    # create the document_store
    document_store = QdrantDocumentStore(
        similarity=similarity,
        index=f"test_similarity_existing_index_{similarity}",
        recreate_index=True,
    )

    # try to connect to the same document store but using the wrong similarity
    non_matching_similarity = "l2" if similarity == "cosine" else "cosine"
    with pytest.raises(
        ValueError,
        match=r"already exists in Qdrant, but it is configured with a similarity .*",
    ):
        document_store2 = QdrantDocumentStore(
            similarity=non_matching_similarity,
            index=f"test_similarity_existing_index_{similarity}",
            recreate_index=False,
        )


@pytest.mark.integration
def test_can_write_id_in_meta(data_store: QdrantDocumentStore):
    document = Document(content="test", meta={"id": "test-id"})
    data_store.write_documents([document])

    documents = data_store.get_all_documents()
    returned_document = documents[0]
    assert returned_document.id == document.id
    assert returned_document.content == document.content
    assert returned_document.meta == document.meta


@pytest.mark.integration
def test_can_write_top_level_fields_in_meta(data_store: QdrantDocumentStore):
    document = Document(content="test", meta={"content": "test-id"})
    data_store.write_documents([document])

    documents = data_store.get_all_documents()
    returned_document = documents[0]
    assert returned_document.id == document.id
    assert returned_document.content == document.content
    assert returned_document.meta == document.meta


@pytest.mark.integration
def test_get_embedding_count(
    data_store: QdrantDocumentStore, documents: List[Document]
):
    data_store.write_documents(documents)
    assert data_store.get_embedding_count() == 9


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_write_labels():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_delete_labels():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_delete_labels_by_id():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_delete_labels_by_filter():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_delete_labels_by_filter_id():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_get_label_count():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_write_labels_duplicate():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_write_get_all_labels():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_labels_with_long_texts():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_multilabel():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_multilabel_no_answer():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_multilabel_filter_aggregations():
    pass


@pytest.mark.skip(reason="Qdrant does not support labels")
@pytest.mark.integration
def test_multilabel_meta_aggregations():
    pass
