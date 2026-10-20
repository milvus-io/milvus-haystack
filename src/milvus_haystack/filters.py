import json
import math
import re
from typing import Any

from haystack.errors import FilterError

LOGIC_OPERATORS = [
    "AND",
    "OR",
    "NOT",
]

COMPARISON_OPERATORS = [
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not in",
]

FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _value_to_str(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return f"[{', '.join(_value_to_str(item) for item in value)}]"
    msg = f"Unsupported filter value type: {type(value).__name__}"
    raise FilterError(msg)


def parse_filters(filters: dict[str, Any]) -> str:
    """
    Parse the filters dictionary into a string which used for milvus expr query.
    :param filters: The filters dictionary.
    :return: The parsed expr.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)
    if "field" in filters:
        return _parse_comparison(filters)
    return _parse_logic(filters)


def _parse_comparison(filters: dict[str, Any]) -> str:
    try:
        _assert_comparison_filter(filters)
    except AssertionError as assert_e:
        raise FilterError(str(assert_e)) from assert_e
    operator = filters["operator"]
    field = filters["field"]
    value = filters["value"]
    if field.startswith("meta."):
        field = field[5:]
    if not FIELD_NAME_PATTERN.fullmatch(field):
        msg = f"Invalid filter field name: {field!r}"
        raise FilterError(msg)
    return f"({field} {operator} {_value_to_str(value)})"


def _assert_comparison_filter(filters: dict[str, Any]):
    assert "operator" in filters, "operator must be specified in filters"  # noqa: S101
    assert "field" in filters, "field must be specified in filters"  # noqa: S101
    assert "value" in filters, "value must be specified in filters"  # noqa: S101
    assert filters["operator"] in COMPARISON_OPERATORS, FilterError(  # noqa: S101
        f"operator must be one of: {COMPARISON_OPERATORS}"
    )


def _parse_logic(filters: dict[str, Any]) -> str:
    try:
        _assert_logic_filter(filters)
    except AssertionError as assert_e:
        raise FilterError(str(assert_e)) from assert_e
    if filters["operator"] == "NOT":  # NOT
        clause_filter = {
            "operator": "AND",
            "conditions": filters["conditions"],
        }
        clause = parse_filters(clause_filter)
        expr = f"not {clause}"
    else:  # AND, OR
        operator = f' {filters["operator"].lower()} '
        expr = operator.join([parse_filters(condition) for condition in filters["conditions"]])
    return f"({expr})"


def _assert_logic_filter(filters: dict[str, Any]):
    assert "operator" in filters, "operator must be specified in filters"  # noqa: S101
    assert "conditions" in filters, "conditions must be specified in filters"  # noqa: S101
    assert filters["operator"] in LOGIC_OPERATORS, f"operator must be one of: {LOGIC_OPERATORS}"  # noqa: S101
    assert isinstance(filters["conditions"], list), "conditions must be a list"  # noqa: S101
