import pytest
import qdrant_client.qdrant_remote

from qdrant_haystack import QdrantDocumentStore


@pytest.mark.integration
def test_passing_metadata_propagates_as_rest_headers():
    doc_store = QdrantDocumentStore("http://localhost:6333", metadata={"foo": "bar"})

    assert doc_store is not None
    assert isinstance(
        doc_store.client._client, qdrant_client.qdrant_remote.QdrantRemote
    )
    assert "foo" in doc_store.client._client._rest_headers
    assert doc_store.client._client._rest_headers["foo"] == "bar"
