from typing import List
from haystack import Document
from haystack.testing.document_store import CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest
from qdrant_haystack.document_stores import QdrantDocumentStore
from haystack.document_stores import DuplicatePolicy
from haystack.document_stores.errors import DuplicateDocumentError

import pytest


class TestQdrantStoreBaseTests(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    @pytest.fixture
    def document_store(self) -> QdrantDocumentStore:
        yield QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
        )

    def assert_documents_are_equal(
        self, received: List[Document], expected: List[Document]
    ):
        """
        Assert that two lists of Documents are equal.
        This is used in every test.
        """

        # Check that the lengths of the lists are the same
        assert len(received) == len(expected)

        # Check that the sets are equal, meaning the content and IDs match regardless of order
        assert set(doc.id for doc in received) == set(doc.id for doc in expected)
        
    def test_write_documents(self, document_store: QdrantDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)