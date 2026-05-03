"""
BigQuery SQL pattern templates for synthetic data generation.
Each template is a (schema_hint, question, sql) triple used to seed LLM generation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SQLTemplate:
    tier: str  # easy | medium | hard
    schema_table: str
    question: str
    sql: str
    tags: list[str]


# ── Easy templates ────────────────────────────────────────────────────────────

EASY: list[SQLTemplate] = [
    SQLTemplate(
        tier="easy",
        schema_table="orders",
        question="How many orders were placed in the last 30 days?",
        sql="""SELECT COUNT(*) AS order_count
FROM `project.ecommerce.orders`
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)""",
        tags=["count", "timestamp_filter"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="orders",
        question="What is the total revenue from delivered orders this year?",
        sql="""SELECT SUM(amount_usd) AS total_revenue_usd
FROM `project.ecommerce.orders`
WHERE status = 'delivered'
  AND EXTRACT(YEAR FROM created_at) = EXTRACT(YEAR FROM CURRENT_DATE())""",
        tags=["sum", "year_filter", "status_filter"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="orders",
        question="For each customer, show their total spend and the date of their most recent order. Only include customers with more than 3 orders.",
        sql="""SELECT
  customer_id,
  SUM(amount_usd)  AS total_spend_usd,
  MAX(created_at)  AS last_order_at
FROM `project.ecommerce.orders`
WHERE status != 'cancelled'
GROUP BY customer_id
HAVING COUNT(*) > 3
ORDER BY total_spend_usd DESC""",
        tags=["group_by", "having", "aggregates"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="customers",
        question="List all active gold and platinum tier customers from the US.",
        sql="""SELECT
  customer_id,
  email,
  first_name,
  last_name,
  tier
FROM `project.ecommerce.customers`
WHERE is_active = TRUE
  AND tier IN ('gold', 'platinum')
  AND country = 'US'
ORDER BY tier, last_name""",
        tags=["filter", "in_clause", "order_by"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="products",
        question="Show the top 10 most expensive active products with their category.",
        sql="""SELECT
  product_id,
  name,
  category,
  price_usd
FROM `project.catalog.products`
WHERE is_active = TRUE
ORDER BY price_usd DESC
LIMIT 10""",
        tags=["order_by", "limit", "filter"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="ad_spend",
        question="What was the total ad spend per channel last month?",
        sql="""SELECT
  channel,
  SUM(spend_usd) AS total_spend_usd
FROM `project.marketing.ad_spend`
WHERE spend_date BETWEEN DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)
                     AND LAST_DAY(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH))
GROUP BY channel
ORDER BY total_spend_usd DESC""",
        tags=["date_trunc", "last_day", "group_by"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="support_tickets",
        question="How many tickets were resolved vs open by category?",
        sql="""SELECT
  category,
  COUNTIF(status = 'resolved') AS resolved_count,
  COUNTIF(status = 'open')     AS open_count,
  COUNT(*)                     AS total_count
FROM `project.support.support_tickets`
GROUP BY category
ORDER BY total_count DESC""",
        tags=["countif", "group_by", "pivot_style"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="inventory",
        question="Which SKUs are below their reorder point as of today's snapshot?",
        sql="""SELECT
  sku,
  warehouse_id,
  quantity_on_hand,
  reorder_point
FROM `project.ops.inventory`
WHERE snapshot_date = CURRENT_DATE()
  AND reorder_point IS NOT NULL
  AND quantity_on_hand < reorder_point
ORDER BY quantity_on_hand ASC""",
        tags=["filter", "null_check", "comparison"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="subscriptions",
        question="How many active subscriptions are there per plan?",
        sql="""SELECT
  plan_name,
  COUNT(*) AS active_subscriptions,
  SUM(mrr_usd) AS total_mrr_usd
FROM `project.billing.subscriptions`
WHERE status = 'active'
GROUP BY plan_name
ORDER BY total_mrr_usd DESC""",
        tags=["group_by", "sum", "filter"],
    ),
    SQLTemplate(
        tier="easy",
        schema_table="returns",
        question="What is the refund rate by return reason in the last 90 days?",
        sql="""SELECT
  reason,
  COUNT(*) AS total_returns,
  COUNTIF(status = 'refunded') AS refunded_count,
  ROUND(COUNTIF(status = 'refunded') / COUNT(*) * 100, 2) AS refund_rate_pct
FROM `project.ecommerce.returns`
WHERE requested_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
GROUP BY reason
ORDER BY total_returns DESC""",
        tags=["countif", "round", "rate_calc"],
    ),
]


# ── Medium templates ──────────────────────────────────────────────────────────

MEDIUM: list[SQLTemplate] = [
    SQLTemplate(
        tier="medium",
        schema_table="orders,customers",
        question="Rank customers by their total spend in the last 6 months and show their tier.",
        sql="""WITH customer_spend AS (
  SELECT
    o.customer_id,
    SUM(o.amount_usd) AS total_spend_usd,
    COUNT(*)          AS order_count
  FROM `project.ecommerce.orders` o
  WHERE o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 180 DAY)
    AND o.status NOT IN ('cancelled', 'refunded')
  GROUP BY o.customer_id
)
SELECT
  cs.customer_id,
  c.email,
  c.tier,
  cs.total_spend_usd,
  cs.order_count,
  RANK() OVER (ORDER BY cs.total_spend_usd DESC) AS spend_rank
FROM customer_spend cs
JOIN `project.ecommerce.customers` c USING (customer_id)
ORDER BY spend_rank""",
        tags=["cte", "join", "rank", "window_function"],
    ),
    SQLTemplate(
        tier="medium",
        schema_table="events",
        question="Calculate the 7-day rolling average of daily page views.",
        sql="""WITH daily_views AS (
  SELECT
    DATE(occurred_at) AS event_date,
    COUNT(*)          AS page_views
  FROM `project.analytics.events`
  WHERE event_type = 'page_view'
  GROUP BY event_date
)
SELECT
  event_date,
  page_views,
  AVG(page_views) OVER (
    ORDER BY event_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7d_avg
FROM daily_views
ORDER BY event_date""",
        tags=["cte", "rolling_avg", "rows_window"],
    ),
    SQLTemplate(
        tier="medium",
        schema_table="orders",
        question="For each order, show the previous order amount for the same customer using LAG.",
        sql="""SELECT
  order_id,
  customer_id,
  created_at,
  amount_usd,
  LAG(amount_usd) OVER (
    PARTITION BY customer_id
    ORDER BY created_at
  ) AS prev_order_amount_usd
FROM `project.ecommerce.orders`
WHERE status NOT IN ('cancelled', 'refunded')
ORDER BY customer_id, created_at""",
        tags=["lag", "partition_by", "window_function"],
    ),
    SQLTemplate(
        tier="medium",
        schema_table="sessions,events",
        question="Find sessions where a purchase event occurred and show session duration.",
        sql="""WITH purchase_sessions AS (
  SELECT DISTINCT session_id
  FROM `project.analytics.events`
  WHERE event_type = 'purchase'
    AND session_id IS NOT NULL
)
SELECT
  s.session_id,
  s.user_id,
  s.started_at,
  s.duration_seconds,
  s.platform,
  s.utm_source
FROM `project.analytics.sessions` s
INNER JOIN purchase_sessions ps USING (session_id)
ORDER BY s.started_at DESC""",
        tags=["cte", "inner_join", "distinct", "semi_join"],
    ),
    SQLTemplate(
        tier="medium",
        schema_table="ad_spend",
        question="Calculate ROAS (return on ad spend) per channel per month for 2024.",
        sql="""SELECT
  FORMAT_DATE('%Y-%m', spend_date)  AS month,
  channel,
  SUM(spend_usd)                    AS total_spend_usd,
  SUM(revenue_usd)                  AS total_revenue_usd,
  ROUND(SUM(revenue_usd) / NULLIF(SUM(spend_usd), 0), 2) AS roas
FROM `project.marketing.ad_spend`
WHERE EXTRACT(YEAR FROM spend_date) = 2024
  AND revenue_usd IS NOT NULL
GROUP BY month, channel
ORDER BY month, roas DESC""",
        tags=["format_date", "nullif", "division_safe", "group_by"],
    ),
    SQLTemplate(
        tier="medium",
        schema_table="subscriptions",
        question="Show month-over-month MRR change using LEAD to compare to the next month.",
        sql="""WITH monthly_mrr AS (
  SELECT
    DATE_TRUNC(started_at, MONTH) AS month,
    SUM(mrr_usd) AS total_mrr
  FROM `project.billing.subscriptions`
  WHERE status = 'active'
  GROUP BY month
)
SELECT
  month,
  total_mrr,
  LEAD(total_mrr) OVER (ORDER BY month) AS next_month_mrr,
  ROUND(
    (LEAD(total_mrr) OVER (ORDER BY month) - total_mrr) / NULLIF(total_mrr, 0) * 100,
    2
  ) AS mom_growth_pct
FROM monthly_mrr
ORDER BY month""",
        tags=["lead", "date_trunc", "mom_growth", "cte"],
    ),
    SQLTemplate(
        tier="medium",
        schema_table="support_tickets",
        question="Find the top 3 agents by average resolution time per category.",
        sql="""WITH agent_stats AS (
  SELECT
    agent_id,
    category,
    AVG(resolution_time_minutes) AS avg_resolution_mins,
    COUNT(*) AS ticket_count
  FROM `project.support.support_tickets`
  WHERE status = 'resolved'
    AND agent_id IS NOT NULL
    AND resolution_time_minutes IS NOT NULL
  GROUP BY agent_id, category
)
SELECT
  category,
  agent_id,
  avg_resolution_mins,
  ticket_count,
  ROW_NUMBER() OVER (PARTITION BY category ORDER BY avg_resolution_mins ASC) AS rank_in_category
FROM agent_stats
QUALIFY ROW_NUMBER() OVER (PARTITION BY category ORDER BY avg_resolution_mins ASC) <= 3
ORDER BY category, rank_in_category""",
        tags=["cte", "row_number", "qualify", "partition_by"],
    ),
    SQLTemplate(
        tier="medium",
        schema_table="orders,returns",
        question="What percentage of orders resulted in a return, grouped by shipping country?",
        sql="""WITH return_counts AS (
  SELECT order_id
  FROM `project.ecommerce.returns`
  WHERE status IN ('approved', 'refunded', 'exchanged')
)
SELECT
  o.shipping_country,
  COUNT(DISTINCT o.order_id)             AS total_orders,
  COUNT(DISTINCT r.order_id)             AS returned_orders,
  ROUND(
    COUNT(DISTINCT r.order_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0),
    2
  ) AS return_rate_pct
FROM `project.ecommerce.orders` o
LEFT JOIN return_counts r USING (order_id)
WHERE o.shipping_country IS NOT NULL
GROUP BY o.shipping_country
ORDER BY return_rate_pct DESC""",
        tags=["cte", "left_join", "distinct_count", "rate_calc"],
    ),
]


# ── Hard templates ────────────────────────────────────────────────────────────

HARD: list[SQLTemplate] = [
    SQLTemplate(
        tier="hard",
        schema_table="orders",
        question="For each customer, return only their single most recent order using QUALIFY.",
        sql="""SELECT
  order_id,
  customer_id,
  created_at,
  amount_usd,
  status
FROM `project.ecommerce.orders`
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) = 1""",
        tags=["qualify", "row_number", "dedup"],
    ),
    SQLTemplate(
        tier="hard",
        schema_table="orders",
        question="Unnest the items array in orders and find the top 5 SKUs by total revenue.",
        sql="""SELECT
  item.sku,
  SUM(item.qty * item.unit_price) AS total_revenue_usd,
  SUM(item.qty)                   AS total_units_sold
FROM `project.ecommerce.orders`,
  UNNEST(items) AS item
WHERE status = 'delivered'
GROUP BY item.sku
ORDER BY total_revenue_usd DESC
LIMIT 5""",
        tags=["unnest", "struct", "cross_join_unnest"],
    ),
    SQLTemplate(
        tier="hard",
        schema_table="orders",
        question="Build a product affinity table: for each pair of SKUs that appear in the same order, count how many orders contain both.",
        sql="""WITH order_skus AS (
  SELECT
    order_id,
    item.sku AS sku
  FROM `project.ecommerce.orders`,
    UNNEST(items) AS item
  WHERE status = 'delivered'
)
SELECT
  a.sku        AS sku_a,
  b.sku        AS sku_b,
  COUNT(*)     AS co_occurrence_count
FROM order_skus a
JOIN order_skus b
  ON a.order_id = b.order_id
  AND a.sku < b.sku
GROUP BY sku_a, sku_b
ORDER BY co_occurrence_count DESC
LIMIT 50""",
        tags=["unnest", "self_join", "affinity", "cte"],
    ),
    SQLTemplate(
        tier="hard",
        schema_table="events",
        question="Aggregate all unique event types seen per user into an ARRAY_AGG, ordered by first occurrence.",
        sql="""SELECT
  user_id,
  ARRAY_AGG(
    DISTINCT event_type
    ORDER BY MIN(occurred_at)
  ) AS event_sequence
FROM (
  SELECT
    user_id,
    event_type,
    MIN(occurred_at) AS first_seen
  FROM `project.analytics.events`
  WHERE user_id IS NOT NULL
  GROUP BY user_id, event_type
)
GROUP BY user_id""",
        tags=["array_agg", "distinct", "subquery"],
    ),
    SQLTemplate(
        tier="hard",
        schema_table="products",
        question="Unnest the attributes array of products and pivot color and size into separate columns.",
        sql="""WITH attrs_flat AS (
  SELECT
    product_id,
    name,
    attr.key,
    attr.value
  FROM `project.catalog.products`,
    UNNEST(attributes) AS attr
  WHERE attr.key IN ('color', 'size')
)
SELECT
  product_id,
  name,
  MAX(IF(key = 'color', value, NULL)) AS color,
  MAX(IF(key = 'size',  value, NULL)) AS size
FROM attrs_flat
GROUP BY product_id, name""",
        tags=["unnest", "struct", "pivot", "conditional_agg"],
    ),
    SQLTemplate(
        tier="hard",
        schema_table="events",
        question="Find users who completed the full funnel: page_view → add_to_cart → purchase, all within 24 hours.",
        sql="""WITH funnel AS (
  SELECT
    user_id,
    MIN(IF(event_type = 'page_view',     occurred_at, NULL)) AS first_page_view,
    MIN(IF(event_type = 'add_to_cart',   occurred_at, NULL)) AS first_add_to_cart,
    MIN(IF(event_type = 'purchase',      occurred_at, NULL)) AS first_purchase
  FROM `project.analytics.events`
  WHERE user_id IS NOT NULL
    AND event_type IN ('page_view', 'add_to_cart', 'purchase')
  GROUP BY user_id
)
SELECT
  user_id,
  first_page_view,
  first_add_to_cart,
  first_purchase,
  TIMESTAMP_DIFF(first_purchase, first_page_view, HOUR) AS hours_to_convert
FROM funnel
WHERE first_page_view    IS NOT NULL
  AND first_add_to_cart  IS NOT NULL
  AND first_purchase     IS NOT NULL
  AND first_add_to_cart  > first_page_view
  AND first_purchase     > first_add_to_cart
  AND TIMESTAMP_DIFF(first_purchase, first_page_view, HOUR) <= 24
ORDER BY hours_to_convert""",
        tags=["funnel", "conditional_min", "timestamp_diff", "cte"],
    ),
    SQLTemplate(
        tier="hard",
        schema_table="orders,customers",
        question="For each customer tier, show the median order value using PERCENTILE_CONT.",
        sql="""SELECT
  c.tier,
  COUNT(DISTINCT o.customer_id)                              AS customer_count,
  COUNT(*)                                                   AS order_count,
  ROUND(AVG(o.amount_usd), 2)                                AS avg_order_usd,
  PERCENTILE_CONT(o.amount_usd, 0.5) OVER (PARTITION BY c.tier) AS median_order_usd
FROM `project.ecommerce.orders` o
JOIN `project.ecommerce.customers` c USING (customer_id)
WHERE o.status NOT IN ('cancelled', 'refunded')
  AND c.tier IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY c.tier ORDER BY o.order_id) = 1""",
        tags=["percentile_cont", "qualify", "analytic_dedup", "join"],
    ),
    SQLTemplate(
        tier="hard",
        schema_table="subscriptions",
        question="Build a monthly cohort retention table: for each signup cohort month, show how many customers were still active N months later.",
        sql="""WITH cohorts AS (
  SELECT
    customer_id,
    DATE_TRUNC(started_at, MONTH) AS cohort_month
  FROM `project.billing.subscriptions`
  WHERE status != 'cancelled'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY started_at) = 1
),
monthly_activity AS (
  SELECT DISTINCT
    customer_id,
    DATE_TRUNC(started_at, MONTH) AS active_month
  FROM `project.billing.subscriptions`
  WHERE status = 'active'
)
SELECT
  c.cohort_month,
  DATE_DIFF(ma.active_month, c.cohort_month, MONTH) AS months_since_start,
  COUNT(DISTINCT ma.customer_id)                     AS active_customers
FROM cohorts c
JOIN monthly_activity ma USING (customer_id)
GROUP BY c.cohort_month, months_since_start
ORDER BY c.cohort_month, months_since_start""",
        tags=["cohort", "qualify", "date_diff", "retention"],
    ),
]


ALL_TEMPLATES = EASY + MEDIUM + HARD


def get_templates_by_tier(tier: str) -> list[SQLTemplate]:
    return [t for t in ALL_TEMPLATES if t.tier == tier]


def get_templates_by_tag(tag: str) -> list[SQLTemplate]:
    return [t for t in ALL_TEMPLATES if tag in t.tags]
