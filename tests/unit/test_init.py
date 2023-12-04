import tempfile

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from qdrant_haystack import QdrantDocumentStore


def test_passing_metadata_propagates_as_rest_headers():
    """
    Test that passing metadata propagates as rest headers. It simply checks if the
    exception thrown by QdrantClient is ResponseHandlingException, as it means that
    the initialization went fine.
    :return:
    """
    with pytest.raises(ResponseHandlingException):
        QdrantDocumentStore("http://localhost:6333", metadata={"foo": "bar"})


def test_delete_qdrant_doc_store_does_not_throw_exceptions():
    """
    Test that deleting QdrantDocumentStore does not throw TypeError exception
    https://github.com/qdrant/qdrant-haystack/issues/29#issuecomment-1708277697
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        document_store = QdrantDocumentStore(
            path=tmpdir,
            index="Document",
            embedding_dim=100,
            recreate_index=True,
            hnsw_config={"m": 16, "ef_construct": 64},
        )
        del document_store
        assert True


def test_passing_api_key():
    """
    Test that passing api key propagates as rest headers. It simply checks if the
    exception thrown by QdrantClient is ResponseHandlingException, as it means that
    the initialization went fine.
    :return:
    """
    with pytest.raises(ResponseHandlingException):
        QdrantDocumentStore("http://localhost:6333", api_key="test")
