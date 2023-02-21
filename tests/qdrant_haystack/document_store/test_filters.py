import pytest

from qdrant_haystack.document_stores.filters import QdrantFilterConverter
from qdrant_client.http import models as rest


@pytest.fixture
def qdrant_converter() -> QdrantFilterConverter:
    return QdrantFilterConverter()


def test_qdrant_filter_converter_none(qdrant_converter):
    converted_filter = qdrant_converter.convert(None)

    assert converted_filter is None


def test_qdrant_filter_converter_empty_dict(qdrant_converter):
    converted_filter = qdrant_converter.convert(dict())
    target_filter = rest.Filter()

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


def test_qdrant_filter_converter_empty_list(qdrant_converter):
    converted_filter = qdrant_converter.convert(list())
    target_filter = rest.Filter()

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[({"type": "article"},), ({"type": {"$eq": "article"}},)],
)
def test_qdrant_filter_converter_comparison_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="type",
                match=rest.MatchValue(value="article"),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[
        ({"item_id": ["item_1", "item_2"]},),
        ({"item_id": {"$in": ["item_1", "item_2"]}},),
    ],
)
def test_qdrant_filter_converter_comparison_in(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        should=[
            rest.FieldCondition(
                key="item_id",
                match=rest.MatchValue(value="item_1"),
            ),
            rest.FieldCondition(
                key="item_id",
                match=rest.MatchValue(value="item_2"),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[({"type": {"$ne": "article"}},)],
)
def test_qdrant_filter_converter_ne_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        must_not=[
            rest.FieldCondition(
                key="type",
                match=rest.MatchValue(value="article"),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[
        ({"item_id": {"$nin": ["item_1", "item_2"]}},),
    ],
)
def test_qdrant_filter_converter_nin_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        must_not=[
            rest.FieldCondition(
                key="item_id",
                match=rest.MatchValue(value="item_1"),
            ),
            rest.FieldCondition(
                key="item_id",
                match=rest.MatchValue(value="item_2"),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[({"type": {"$gt": 1.0}},)],
)
def test_qdrant_filter_converter_gt_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="type",
                range=rest.Range(gt=1.0),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[({"type": {"$gte": 2.0}},)],
)
def test_qdrant_filter_converter_gte_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="type",
                range=rest.Range(gte=2.0),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[({"type": {"$lt": 3.0}},)],
)
def test_qdrant_filter_converter_lt_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="type",
                range=rest.Range(lt=3.0),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[({"type": {"$lte": 4.0}},)],
)
def test_qdrant_filter_converter_lte_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        must=[
            rest.FieldCondition(
                key="type",
                range=rest.Range(lte=4.0),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


def test_qdrant_filter_converter_has_id(qdrant_converter):
    converted_filter = qdrant_converter.convert(None, [1, 2, 3])
    target_filter = rest.Filter(
        must=[
            rest.HasIdCondition(
                has_id=[1, 2, 3],
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[({"$not": {"field_name": 212}},)],
)
def test_qdrant_filter_converter_not_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        must_not=[
            rest.FieldCondition(
                key="field_name",
                match=rest.MatchValue(value=212),
            )
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter


@pytest.mark.parametrize(
    argnames=["filter_term"],
    argvalues=[({"$or": [{"field_name": 212}, {"field_name": 211}]},)],
)
def test_qdrant_filter_converter_or_operation(qdrant_converter, filter_term):
    converted_filter = qdrant_converter.convert(filter_term)
    target_filter = rest.Filter(
        should=[
            rest.FieldCondition(
                key="field_name",
                match=rest.MatchValue(value=212),
            ),
            rest.FieldCondition(
                key="field_name",
                match=rest.MatchValue(value=211),
            ),
        ]
    )

    assert converted_filter is not None
    assert isinstance(converted_filter, rest.Filter)
    assert target_filter == converted_filter
