from typing import List
from haystack import Document
from haystack.testing.document_store import (
    TEST_EMBEDDING_1,
    TEST_EMBEDDING_2,
    AssertDocumentsEqualMixin,
    FilterableDocsFixtureMixin,
    LegacyFilterDocumentsInvalidFiltersTest
)
import pandas as pd
import pytest

from haystack.document_stores import DocumentStore
from haystack.utils.filters import FilterError

from qdrant_haystack.document_stores.qdrant import QdrantDocumentStore


class LegacyFilterDocumentsEqualTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using implicit and explicit '$eq' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsEqualTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_filter_document_content(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"content": "A Foo Document 1"}
        )
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.content == "A Foo Document 1"],
        )

    @pytest.mark.unit
    def test_filter_simple_metadata_value(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": "100"})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"]
        )

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_filter_document_dataframe(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"dataframe": pd.DataFrame([1])}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if doc.dataframe is not None and doc.dataframe.equals(pd.DataFrame([1]))
            ],
        )

    @pytest.mark.unit
    def test_eq_filter_explicit(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": {"$eq": "100"}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"]
        )

    @pytest.mark.unit
    def test_eq_filter_implicit(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": "100"})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"]
        )

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    @pytest.mark.unit
    def test_eq_filter_table(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"dataframe": pd.DataFrame([1])}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if isinstance(doc.dataframe, pd.DataFrame)
                and doc.dataframe.equals(pd.DataFrame([1]))
            ],
        )

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    @pytest.mark.unit
    def test_eq_filter_embedding(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        embedding = [0.0] * 768
        result = document_store.filter_documents(filters={"embedding": embedding})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if embedding == doc.embedding]
        )


class LegacyFilterDocumentsNotEqualTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$ne' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsNotEqualTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_ne_filter(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": {"$ne": "100"}})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") != "100"]
        )

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_ne_filter_table(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"dataframe": {"$ne": pd.DataFrame([1])}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if not isinstance(doc.dataframe, pd.DataFrame)
                or not doc.dataframe.equals(pd.DataFrame([1]))
            ],
        )

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_ne_filter_embedding(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"embedding": {"$ne": TEST_EMBEDDING_1}}
        )
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.embedding != TEST_EMBEDDING_1],
        )


class LegacyFilterDocumentsInTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using implicit and explicit '$in' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsInTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_filter_simple_list_single_element(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["100"]})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"]
        )

    @pytest.mark.unit
    def test_filter_simple_list_one_value(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["100"]})
        self.assert_documents_are_equal(
            result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100"]]
        )

    @pytest.mark.unit
    def test_filter_simple_list(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["100", "123"]})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]],
        )

    @pytest.mark.unit
    def test_incorrect_filter_name(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"non_existing_meta_field": ["whatever"]}
        )
        self.assert_documents_are_equal(result, [])

    @pytest.mark.unit
    def test_incorrect_filter_value(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["nope"]})
        self.assert_documents_are_equal(result, [])

    @pytest.mark.unit
    def test_in_filter_explicit(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"meta.page": {"$in": ["100", "123", "n.a."]}}
        )
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]],
        )

    @pytest.mark.unit
    def test_in_filter_implicit(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"meta.page": ["100", "123", "n.a."]}
        )
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]],
        )

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_in_filter_table(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"dataframe": {"$in": [pd.DataFrame([1]), pd.DataFrame([2])]}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if isinstance(doc.dataframe, pd.DataFrame)
                and (
                    doc.dataframe.equals(pd.DataFrame([1]))
                    or doc.dataframe.equals(pd.DataFrame([2]))
                )
            ],
        )

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_in_filter_embedding(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        embedding_zero = [0.0] * 768
        embedding_one = [1.0] * 768
        result = document_store.filter_documents(
            filters={"embedding": {"$in": [embedding_zero, embedding_one]}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (embedding_zero == doc.embedding or embedding_one == doc.embedding)
            ],
        )


class LegacyFilterDocumentsNotInTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$nin' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsNotInTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_nin_filter_table(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"dataframe": {"$nin": [pd.DataFrame([1]), pd.DataFrame([0])]}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if not isinstance(doc.dataframe, pd.DataFrame)
                or (
                    not doc.dataframe.equals(pd.DataFrame([1]))
                    and not doc.dataframe.equals(pd.DataFrame([0]))
                )
            ],
        )

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_nin_filter_embedding(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"embedding": {"$nin": [TEST_EMBEDDING_1, TEST_EMBEDDING_2]}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if doc.embedding not in [TEST_EMBEDDING_1, TEST_EMBEDDING_2]
            ],
        )

    @pytest.mark.unit
    def test_nin_filter(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"meta.page": {"$nin": ["100", "123", "n.a."]}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if doc.meta.get("page") not in ["100", "123"]
            ],
        )


class LegacyFilterDocumentsGreaterThanTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$gt' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsGreaterThanTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_gt_filter(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$gt": 0.0}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] > 0
            ],
        )

    @pytest.mark.unit
    def test_gt_filter_non_numeric(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"meta.page": {"$gt": "100"}})

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_gt_filter_table(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"dataframe": {"$gt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}}
            )

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_gt_filter_embedding(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"embedding": {"$gt": TEST_EMBEDDING_1}}
            )


class LegacyFilterDocumentsGreaterThanEqualTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$gte' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsGreaterThanEqualTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_gte_filter(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$gte": -2}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] >= -2
            ],
        )

    @pytest.mark.unit
    def test_gte_filter_non_numeric(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"page": {"$gte": "100"}})

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_gte_filter_table(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"dataframe": {"$gte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}}
            )

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_gte_filter_embedding(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"embedding": {"$gte": TEST_EMBEDDING_1}}
            )


class LegacyFilterDocumentsLessThanTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$lt' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsLessThanTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_lt_filter(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$lt": 0.0}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if doc.meta.get("number") is not None and doc.meta["number"] < 0
            ],
        )

    @pytest.mark.unit
    def test_lt_filter_non_numeric(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"meta.page": {"$lt": "100"}})

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_lt_filter_table(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"dataframe": {"$lt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}}
            )

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_lt_filter_embedding(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"embedding": {"$lt": TEST_EMBEDDING_2}}
            )


class LegacyFilterDocumentsLessThanEqualTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using explicit '$lte' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsLessThanEqualTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_lte_filter(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$lte": 2.0}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if doc.meta.get("number") is not None and doc.meta["number"] <= 2.0
            ],
        )

    @pytest.mark.unit
    def test_lte_filter_non_numeric(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"meta.page": {"$lte": "100"}})

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_lte_filter_table(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"dataframe": {"$lte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}}
            )

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_lte_filter_embedding(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"embedding": {"$lte": TEST_EMBEDDING_1}}
            )


class LegacyFilterDocumentsSimpleLogicalTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using logical '$and', '$or' and '$not' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsSimpleLogicalTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_filter_simple_or(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters = {
            "$or": {"meta.name": {"$in": ["name_0", "name_1"]}, "meta.number": {"$lt": 1.0}}
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                or doc.meta.get("name") in ["name_0", "name_1"]
            ],
        )

    @pytest.mark.unit
    def test_filter_simple_implicit_and_with_multi_key_dict(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"meta.number": {"$lte": 2.0, "$gte": 0.0}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta
                and doc.meta["number"] >= 0.0
                and doc.meta["number"] <= 2.0
            ],
        )

    @pytest.mark.unit
    def test_filter_simple_explicit_and_with_list(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"meta.number": {"$and": [{"$lte": 2}, {"$gte": 0}]}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta
                and doc.meta["number"] <= 2.0
                and doc.meta["number"] >= 0.0
            ],
        )

    @pytest.mark.unit
    def test_filter_simple_implicit_and(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={"meta.number": {"$lte": 2.0, "$gte": 0}}
        )
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta
                and doc.meta["number"] <= 2.0
                and doc.meta["number"] >= 0.0
            ],
        )


class LegacyFilterDocumentsNestedLogicalTest(
    AssertDocumentsEqualMixin, FilterableDocsFixtureMixin
):
    """
    Utility class to test a Document Store `filter_documents` method using multiple nested logical '$and', '$or' and '$not' legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsNestedLogicalTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_filter_nested_implicit_and(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "meta.number": {"$lte": 2, "$gte": 0},
            "meta.name": ["name_0", "name_1"],
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    "number" in doc.meta
                    and doc.meta["number"] <= 2
                    and doc.meta["number"] >= 0
                    and doc.meta.get("name") in ["name_0", "name_1"]
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_or(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters = {
            "$or": {
                "meta.name": {"$in": ["name_0", "name_1"]},
                "meta.number": {"$lt": 1.0},
            }
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("name") in ["name_0", "name_1"]
                    or (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_and_or_explicit(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "$and": {
                "meta.page": {"$eq": "123"},
                "$or": {
                    "meta.name": {"$in": ["name_0", "name_1"]},
                    "meta.number": {"$lt": 1.0},
                },
            }
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("page") in ["123"]
                    and (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.meta and doc.meta["number"] < 1)
                    )
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_and_or_implicit(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "meta.page": {"$eq": "123"},
            "$or": {
                "meta.name": {"$in": ["name_0", "name_1"]},
                "meta.number": {"$lt": 1.0},
            },
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("page") in ["123"]
                    and (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.meta and doc.meta["number"] < 1)
                    )
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_or_and(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "$or": {
                "meta.number": {"$lt": 1},
                "$and": {
                    "meta.name": {"$in": ["name_0", "name_1"]},
                    "$not": {"meta.chapter": {"$eq": "intro"}},
                },
            }
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                    or (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        and (doc.meta.get("chapter") != "intro")
                    )
                )
            ],
        )

    @pytest.mark.unit
    def test_filter_nested_multiple_identical_operators_same_level(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters = {
            "$or": [
                {
                    "$and": {
                        "meta.name": {"$in": ["name_0", "name_1"]},
                        "meta.page": "100",
                    }
                },
                {
                    "$and": {
                        "meta.chapter": {"$in": ["intro", "abstract"]},
                        "meta.page": "123",
                    }
                },
            ]
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        and doc.meta.get("page") == "100"
                    )
                    or (
                        doc.meta.get("chapter") in ["intro", "abstract"]
                        and doc.meta.get("page") == "123"
                    )
                )
            ],
        )


class TestLegacyFilterDocuments(  # pylint: disable=too-many-ancestors
    LegacyFilterDocumentsInvalidFiltersTest,
    LegacyFilterDocumentsEqualTest,
    LegacyFilterDocumentsNotEqualTest,
    LegacyFilterDocumentsInTest,
    LegacyFilterDocumentsNotInTest,
    LegacyFilterDocumentsGreaterThanTest,
    LegacyFilterDocumentsGreaterThanEqualTest,
    LegacyFilterDocumentsLessThanTest,
    LegacyFilterDocumentsLessThanEqualTest,
    LegacyFilterDocumentsSimpleLogicalTest,
    LegacyFilterDocumentsNestedLogicalTest,
):
    """
    Utility class to test a Document Store `filter_documents` method using different types of legacy filters

    To use it create a custom test class and override the `document_store` fixture to return your Document Store.
    Example usage:

    ```python
    class MyDocumentStoreTest(LegacyFilterDocumentsTest):
        @pytest.fixture
        def document_store(self):
            return MyDocumentStore()
    ```
    """

    @pytest.mark.unit
    def test_no_filter_empty(self, document_store: DocumentStore):
        assert document_store.filter_documents() == []
        assert document_store.filter_documents(filters={}) == []

    @pytest.mark.unit
    def test_no_filter_not_empty(self, document_store: DocumentStore):
        docs = [Document(content="test doc")]
        document_store.write_documents(docs)
        self.assert_documents_are_equal(document_store.filter_documents(), docs)
        self.assert_documents_are_equal(document_store.filter_documents(filters={}), docs)
        
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
