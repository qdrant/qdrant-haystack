from abc import ABC
from typing import Any, List, Optional, Union

from qdrant_client.http import models as rest

from qdrant_haystack.document_stores.converters import HaystackToQdrant


class BaseFilterConverter(ABC):
    """Converts Haystack filters to a format accepted by an external tool."""

    def convert(
        self,
        filter_term: Union[dict, List[dict]],
        allowed_ids: Optional[List[Any]] = None,
    ) -> Optional[Any]:
        raise NotImplementedError


class QdrantFilterConverter(BaseFilterConverter):
    """Converts Haystack filters to the format used by Qdrant."""

    def __init__(self):
        self.haystack_to_qdrant_converter = HaystackToQdrant()

    def convert(
        self,
        filter_term: Optional[Union[dict, List[dict]]] = None,
        allowed_ids: Optional[List[str]] = None,
    ) -> Optional[rest.Filter]:
        if filter_term is None and allowed_ids is None:
            return None

        must_clauses, should_clauses, must_not_clauses = [], [], []
        if allowed_ids is not None:
            must_clauses.append(self._build_has_id_condition(allowed_ids))

        if isinstance(filter_term, dict):
            filter_term = [filter_term]
        if filter_term is None:
            filter_term = []

        for item in filter_term:
            for key, value in item.items():
                if key == "$and":
                    must_clauses.append(self._parse_and(key, value))
                elif key == "$or":
                    should_clauses.append(self._parse_or(key, value))
                elif key == "$not":
                    must_not_clauses.append(self._parse_not(key, value))
                # Key needs to be a payload field
                else:
                    must_clauses.extend(self._parse_comparison_operation(key, value))

        payload_filter = rest.Filter(
            must=must_clauses if len(must_clauses) > 0 else None,
            should=should_clauses if len(should_clauses) > 0 else None,
            must_not=must_not_clauses if len(must_not_clauses) > 0 else None,
        )
        return self._squueze_filter(payload_filter)

    def _parse_not(self, key: str, value: Union[dict, List[dict]]) -> rest.Condition:
        return self.convert(value)

    def _parse_and(self, key: str, value: Union[dict, List[dict]]) -> rest.Condition:
        return self.convert(value)

    def _parse_or(self, key: str, value: Union[dict, List[dict]]) -> rest.Condition:
        return self.convert(value)

    def _parse_comparison_operation(
        self, key: str, value: Union[dict, List, str, float]
    ) -> List[rest.Condition]:
        conditions: List[rest.Condition] = []

        if isinstance(value, dict):
            for comparison_operation, comparison_value in value.items():
                if comparison_operation == "$eq":
                    conditions.append(self._build_eq_condition(key, comparison_value))
                elif comparison_operation == "$in":
                    conditions.append(self._build_in_condition(key, comparison_value))
                elif comparison_operation == "$ne":
                    conditions.append(self._build_ne_condition(key, comparison_value))
                elif comparison_operation == "$nin":
                    conditions.append(self._build_nin_condition(key, comparison_value))
                elif comparison_operation == "$gt":
                    conditions.append(self._build_gt_condition(key, comparison_value))
                elif comparison_operation == "$gte":
                    conditions.append(self._build_gte_condition(key, comparison_value))
                elif comparison_operation == "$lt":
                    conditions.append(self._build_lt_condition(key, comparison_value))
                elif comparison_operation == "$lte":
                    conditions.append(self._build_lte_condition(key, comparison_value))
                else:
                    raise ValueError(
                        f"Unknown operator {comparison_operation} used in filters"
                    )
        elif isinstance(value, list):
            conditions.append(self._build_in_condition(key, value))
        else:
            conditions.append(self._build_eq_condition(key, value))

        return conditions

    def _build_eq_condition(
        self, key: str, value: rest.ValueVariants
    ) -> rest.Condition:
        return rest.FieldCondition(
            key=f"meta.{key}", match=rest.MatchValue(value=value)
        )

    def _build_in_condition(
        self, key: str, value: List[rest.ValueVariants]
    ) -> rest.Condition:
        return rest.Filter(
            should=[
                rest.FieldCondition(
                    key=f"meta.{key}", match=rest.MatchValue(value=item)
                )
                for item in value
            ]
        )

    def _build_ne_condition(
        self, key: str, value: rest.ValueVariants
    ) -> rest.Condition:
        return rest.Filter(
            must_not=[
                rest.FieldCondition(
                    key=f"meta.{key}", match=rest.MatchValue(value=value)
                )
            ]
        )

    def _build_nin_condition(
        self, key: str, value: List[rest.ValueVariants]
    ) -> rest.Condition:
        return rest.Filter(
            must_not=[
                rest.FieldCondition(
                    key=f"meta.{key}", match=rest.MatchValue(value=item)
                )
                for item in value
            ]
        )

    def _build_lt_condition(
        self, key: str, value: rest.ValueVariants
    ) -> rest.Condition:
        return rest.FieldCondition(key=f"meta.{key}", range=rest.Range(lt=value))

    def _build_lte_condition(
        self, key: str, value: rest.ValueVariants
    ) -> rest.Condition:
        return rest.FieldCondition(key=f"meta.{key}", range=rest.Range(lte=value))

    def _build_gt_condition(
        self, key: str, value: rest.ValueVariants
    ) -> rest.Condition:
        return rest.FieldCondition(key=f"meta.{key}", range=rest.Range(gt=value))

    def _build_gte_condition(
        self, key: str, value: rest.ValueVariants
    ) -> rest.Condition:
        return rest.FieldCondition(key=f"meta.{key}", range=rest.Range(gte=value))

    def _build_has_id_condition(
        self, id_values: List[rest.ExtendedPointId]
    ) -> rest.HasIdCondition:
        return rest.HasIdCondition(
            has_id=[
                # Ids are converted into their internal representation
                self.haystack_to_qdrant_converter.convert_id(item)
                for item in id_values
            ]
        )

    def _squueze_filter(self, payload_filter: rest.Filter) -> rest.Filter:
        """
        Simplify given payload filter, if the nested structure might be unnested.
        That happens if there is a single clause in that filter.
        :param payload_filter:
        :return:
        """
        filter_parts = {
            "must": payload_filter.must,
            "should": payload_filter.should,
            "must_not": payload_filter.must_not,
        }

        total_clauses = sum(len(x) for x in filter_parts.values() if x is not None)
        if total_clauses == 0 or total_clauses > 1:
            return payload_filter

        # Payload filter has just a single clause provided (either must, should
        # or must_not). If that single clause is also of a rest.Filter type,
        # then it might be returned instead.
        for part_name, filter_part in filter_parts.items():
            if filter_part is None or 0 == len(filter_part):
                continue

            subfilter = filter_part[0]
            if not isinstance(subfilter, rest.Filter):
                # The inner statement is a simple condition like rest.FieldCondition
                # so it cannot be simplified.
                continue

            if "must" == part_name:
                # If the parent is a must statement, then we may just return
                # the inner one. For should and must_not that has to be handled
                # differently.
                return subfilter

            if subfilter.must is not None and len(subfilter.must) > 0:
                return rest.Filter(**{part_name: subfilter.must})

        return payload_filter
