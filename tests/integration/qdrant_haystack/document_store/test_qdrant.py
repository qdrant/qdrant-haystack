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
    """
    Weaviate doesn't include documents if the field is missing,
    so we customize this test
    """
    data_store.write_documents(documents)

    result = data_store.get_all_documents(
        filters={"year": {"$nin": ["2020", "2021", "n.a."]}}
    )
    assert len(result) == 0


@pytest.mark.integration
def test_delete_index(data_store: QdrantDocumentStore, documents: List[Document]):
    """Contrary to other Document Stores, this doesn't raise if the index is empty"""
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
def test_query(data_store: QdrantDocumentStore, documents: List[Document]):
    data_store.write_documents(documents)

    query_text = "Foo"
    docs = data_store.query(query_text)
    assert len(docs) == 3

    # BM25 retrieval WITH filters is not yet supported as of Weaviate v1.14.1
    # Should be from 1.18: https://github.com/semi-technologies/weaviate/issues/2393
    # docs = ds.query(query_text, filters={"name": ["name_1"]})
    # assert len(docs) == 1

    docs = data_store.query(query=None, filters={"name": ["name_0"]})
    assert len(docs) == 3

    docs = data_store.query(query=None, filters={"content": [query_text.lower()]})
    assert len(docs) == 3

    docs = data_store.query(query=None, filters={"content": ["baz"]})
    assert len(docs) == 3


@pytest.mark.integration
def test_get_all_documents_unaffected_by_QUERY_MAXIMUM_RESULTS(
    data_store: QdrantDocumentStore, documents: List[Document], monkeypatch
):
    """
    Ensure `get_all_documents` works no matter the value of QUERY_MAXIMUM_RESULTS
    see https://github.com/deepset-ai/haystack/issues/2517
    """
    data_store.write_documents(documents)
    monkeypatch.setattr(data_store, "get_document_count", lambda **kwargs: 13_000)
    docs = data_store.get_all_documents()
    assert len(docs) == 9


@pytest.mark.integration
def test_deleting_by_id_or_by_filters(
    data_store: QdrantDocumentStore, documents: List[Document]
):
    data_store.write_documents(documents)
    # This test verifies that deleting an object by its ID does not first require fetching all documents. This fixes
    # a bug, as described in https://github.com/deepset-ai/haystack/issues/2898
    data_store.get_all_documents = MagicMock(wraps=data_store.get_all_documents)

    assert data_store.get_document_count() == 9

    # Delete a document by its ID. This should bypass the get_all_documents() call
    data_store.delete_documents(ids=[documents[0].id])
    data_store.get_all_documents.assert_not_called()
    assert data_store.get_document_count() == 8

    data_store.get_all_documents.reset_mock()
    # Delete a document with filters. Prove that using the filters will go through get_all_documents()
    data_store.delete_documents(filters={"name": ["name_0"]})
    data_store.get_all_documents.assert_called()
    assert data_store.get_document_count() == 6


@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["cosine", "l2", "dot_product"])
def test_similarity_existing_index(similarity):
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
        ValueError, match=r"This index already exists in Qdrant with similarity .*"
    ):
        document_store2 = QdrantDocumentStore(
            similarity=non_matching_similarity,
            index=f"test_similarity_existing_index_{similarity}",
            recreate_index=False,
        )


@pytest.mark.integration
def test_cant_write_id_in_meta(data_store: QdrantDocumentStore):
    with pytest.raises(ValueError, match='"meta" info contains duplicate key "id"'):
        data_store.write_documents([Document(content="test", meta={"id": "test-id"})])


@pytest.mark.integration
def test_cant_write_top_level_fields_in_meta(data_store: QdrantDocumentStore):
    with pytest.raises(
        ValueError, match='"meta" info contains duplicate key "content"'
    ):
        data_store.write_documents(
            [Document(content="test", meta={"content": "test-id"})]
        )


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
