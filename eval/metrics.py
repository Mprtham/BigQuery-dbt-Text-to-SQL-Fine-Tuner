"""
Evaluation metrics: Exact Match and sqlglot BigQuery parse accuracy.
"""

import re
import json
from pathlib import Path
from typing import Optional

import sqlglot
import sqlglot.errors


def normalise_sql(sql: str) -> str:
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"[`'\"]", "", sql)
    sql = re.sub(r"\s*,\s*", ",", sql)
    sql = re.sub(r"\s*=\s*", "=", sql)
    return sql


def exact_match(pred: str, gold: str) -> bool:
    return normalise_sql(pred) == normalise_sql(gold)


def partial_match(pred: str, gold: str, threshold: float = 0.8) -> float:
    pred_tokens = set(normalise_sql(pred).split())
    gold_tokens = set(normalise_sql(gold).split())
    if not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    return overlap / len(gold_tokens)


def bq_parse_valid(sql: str) -> tuple[bool, Optional[str]]:
    """Returns (is_valid, error_message). Uses sqlglot BigQuery dialect."""
    # Strip dbt Jinja for parse check
    sql_stripped = re.sub(r"\{\{.*?\}\}", "'__jinja__'", sql, flags=re.DOTALL)
    sql_stripped = re.sub(r"\{%.*?%\}", "", sql_stripped, flags=re.DOTALL)
    sql_stripped = sql_stripped.strip()
    if not sql_stripped:
        return True, None  # dbt-only template, skip parse
    try:
        sqlglot.parse(sql_stripped, dialect="bigquery", error_level=sqlglot.ErrorLevel.RAISE)
        return True, None
    except sqlglot.errors.ParseError as e:
        return False, str(e)


def schema_compliance(sql: str, schema_columns: list[str]) -> float:
    """What fraction of schema columns used in the SQL actually exist in schema."""
    if not schema_columns:
        return 1.0
    normalised_cols = [c.lower() for c in schema_columns]
    sql_lower = sql.lower()
    referenced = [col for col in normalised_cols if col in sql_lower]
    total_in_sql = sum(1 for col in normalised_cols if col in sql_lower)
    if total_in_sql == 0:
        return 1.0
    return len(referenced) / len(normalised_cols)


def evaluate_batch(
    predictions: list[str],
    gold_sqls: list[str],
    schema_columns_list: Optional[list[list[str]]] = None,
) -> dict:
    n = len(predictions)
    em_scores       = []
    partial_scores  = []
    parse_valid     = []
    schema_scores   = []

    for i, (pred, gold) in enumerate(zip(predictions, gold_sqls)):
        em_scores.append(int(exact_match(pred, gold)))
        partial_scores.append(partial_match(pred, gold))
        valid, _ = bq_parse_valid(pred)
        parse_valid.append(int(valid))
        if schema_columns_list:
            schema_scores.append(schema_compliance(pred, schema_columns_list[i]))

    results = {
        "n":                    n,
        "exact_match":          sum(em_scores) / n,
        "partial_match_avg":    sum(partial_scores) / n,
        "parse_accuracy":       sum(parse_valid) / n,
    }
    if schema_scores:
        results["schema_compliance"] = sum(schema_scores) / n

    return results


def print_metrics(results: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}n={results['n']}")
    print(f"{prefix}Exact Match:       {results['exact_match']:.1%}")
    print(f"{prefix}Partial Match avg: {results['partial_match_avg']:.1%}")
    print(f"{prefix}Parse Accuracy:    {results['parse_accuracy']:.1%}")
    if "schema_compliance" in results:
        print(f"{prefix}Schema Compliance: {results['schema_compliance']:.1%}")
