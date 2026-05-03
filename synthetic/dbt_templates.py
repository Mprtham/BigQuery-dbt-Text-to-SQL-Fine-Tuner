"""
dbt Jinja SQL pattern templates for synthetic data generation.
Covers ref(), source(), is_incremental(), config(), and dbt_utils macros.
"""

from dataclasses import dataclass


@dataclass
class DBTTemplate:
    tier: str
    pattern: str  # jinja_pattern | macro | incremental | config
    question: str
    sql: str
    tags: list[str]


DBT_TEMPLATES: list[DBTTemplate] = [
    # ── ref() and source() ─────────────────────────────────────────────────────
    DBTTemplate(
        tier="dbt",
        pattern="ref",
        question="Write a dbt model that joins the stg_orders and stg_customers staging models to produce a mart-level order enriched view.",
        sql="""{{
  config(
    materialized = 'table',
    tags         = ['mart', 'orders']
  )
}}

WITH orders AS (
  SELECT * FROM {{ ref('stg_orders') }}
),
customers AS (
  SELECT * FROM {{ ref('stg_customers') }}
)

SELECT
  o.order_id,
  o.customer_id,
  o.created_at,
  o.amount_usd,
  o.status,
  c.email,
  c.tier           AS customer_tier,
  c.country        AS customer_country,
  c.acquisition_channel
FROM orders o
LEFT JOIN customers c USING (customer_id)""",
        tags=["ref", "config", "join", "mart"],
    ),
    DBTTemplate(
        tier="dbt",
        pattern="source",
        question="Write a dbt staging model that reads raw orders from the source and normalises status to lowercase.",
        sql="""{{
  config(
    materialized = 'view',
    tags         = ['staging', 'orders']
  )
}}

SELECT
  order_id,
  customer_id,
  created_at,
  updated_at,
  LOWER(status)                        AS status,
  COALESCE(amount_usd, 0.0)            AS amount_usd,
  COALESCE(discount_usd, 0.0)          AS discount_usd,
  shipping_country,
  items,
  tags
FROM {{ source('ecommerce', 'orders') }}""",
        tags=["source", "config", "coalesce", "staging"],
    ),
    # ── is_incremental() ───────────────────────────────────────────────────────
    DBTTemplate(
        tier="dbt",
        pattern="incremental",
        question="Write an incremental dbt model that appends new events from the raw events source table, using event_id as the unique key.",
        sql="""{{
  config(
    materialized       = 'incremental',
    unique_key         = 'event_id',
    incremental_strategy = 'merge',
    tags               = ['events', 'incremental']
  )
}}

SELECT
  event_id,
  user_id,
  anonymous_id,
  event_type,
  occurred_at,
  session_id,
  platform,
  properties,
  page_url,
  country
FROM {{ source('analytics', 'events') }}

{% if is_incremental() %}
  WHERE occurred_at > (SELECT MAX(occurred_at) FROM {{ this }})
{% endif %}""",
        tags=["incremental", "is_incremental", "this", "merge"],
    ),
    DBTTemplate(
        tier="dbt",
        pattern="incremental",
        question="Write an incremental dbt model for daily ad spend aggregates that uses insert_overwrite partitioned by spend_date.",
        sql="""{{
  config(
    materialized         = 'incremental',
    incremental_strategy = 'insert_overwrite',
    partition_by         = {'field': 'spend_date', 'data_type': 'date'},
    tags                 = ['marketing', 'incremental']
  )
}}

SELECT
  spend_date,
  channel,
  campaign_id,
  campaign_name,
  SUM(impressions)  AS impressions,
  SUM(clicks)       AS clicks,
  SUM(spend_usd)    AS spend_usd,
  SUM(conversions)  AS conversions,
  SUM(revenue_usd)  AS revenue_usd
FROM {{ source('marketing', 'ad_spend') }}

{% if is_incremental() %}
  WHERE spend_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)
{% endif %}

GROUP BY spend_date, channel, campaign_id, campaign_name""",
        tags=["incremental", "partition_by", "insert_overwrite", "aggregation"],
    ),
    # ── dbt_utils macros ───────────────────────────────────────────────────────
    DBTTemplate(
        tier="dbt",
        pattern="macro",
        question="Write a dbt model that generates a daily date spine for the last 365 days using dbt_utils.date_spine.",
        sql="""{{
  config(
    materialized = 'table',
    tags         = ['utility', 'date_spine']
  )
}}

WITH date_spine AS (
  {{ dbt_utils.date_spine(
      datepart   = 'day',
      start_date = "DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)",
      end_date   = "CURRENT_DATE()"
  ) }}
)

SELECT
  date_day                                 AS calendar_date,
  EXTRACT(YEAR  FROM date_day)             AS year,
  EXTRACT(MONTH FROM date_day)             AS month,
  EXTRACT(DAY   FROM date_day)             AS day_of_month,
  EXTRACT(DAYOFWEEK FROM date_day)         AS day_of_week,
  FORMAT_DATE('%A', date_day)              AS day_name,
  DATE_TRUNC(date_day, WEEK)               AS week_start,
  DATE_TRUNC(date_day, MONTH)              AS month_start,
  DATE_TRUNC(date_day, QUARTER)            AS quarter_start
FROM date_spine""",
        tags=["dbt_utils", "date_spine", "date_dim"],
    ),
    DBTTemplate(
        tier="dbt",
        pattern="macro",
        question="Use dbt_utils.generate_surrogate_key to create a surrogate key for the orders mart from order_id and customer_id.",
        sql="""{{
  config(
    materialized = 'table',
    tags         = ['mart', 'keys']
  )
}}

SELECT
  {{ dbt_utils.generate_surrogate_key(['order_id', 'customer_id']) }} AS surrogate_key,
  order_id,
  customer_id,
  created_at,
  amount_usd,
  status
FROM {{ ref('stg_orders') }}""",
        tags=["dbt_utils", "surrogate_key", "mart"],
    ),
    DBTTemplate(
        tier="dbt",
        pattern="macro",
        question="Write a dbt model using dbt_utils.union_relations to union together three regional orders tables.",
        sql="""{{
  config(
    materialized = 'table',
    tags         = ['mart', 'union']
  )
}}

{{
  dbt_utils.union_relations(
    relations = [
      ref('stg_orders_us'),
      ref('stg_orders_eu'),
      ref('stg_orders_apac')
    ]
  )
}}""",
        tags=["dbt_utils", "union_relations", "multi_source"],
    ),
    # ── config() patterns ──────────────────────────────────────────────────────
    DBTTemplate(
        tier="dbt",
        pattern="config",
        question="Write a dbt model with cluster_by on customer_id and partition_by on created_at date for BigQuery performance.",
        sql="""{{
  config(
    materialized = 'table',
    partition_by = {
      'field'       : 'created_at',
      'data_type'   : 'timestamp',
      'granularity' : 'day'
    },
    cluster_by   = ['customer_id', 'status'],
    tags         = ['mart', 'orders', 'optimised']
  )
}}

SELECT
  order_id,
  customer_id,
  created_at,
  DATE(created_at)  AS order_date,
  amount_usd,
  status,
  shipping_country
FROM {{ ref('stg_orders') }}
WHERE status IS NOT NULL""",
        tags=["config", "partition_by", "cluster_by", "bq_performance"],
    ),
    DBTTemplate(
        tier="dbt",
        pattern="config",
        question="Write a dbt snapshot model to track slowly-changing customer tier over time.",
        sql="""{% snapshot customers_snapshot %}

{{
  config(
    target_schema  = 'snapshots',
    unique_key     = 'customer_id',
    strategy       = 'check',
    check_cols     = ['tier', 'is_active', 'lifetime_value_usd']
  )
}}

SELECT
  customer_id,
  email,
  tier,
  is_active,
  lifetime_value_usd,
  country
FROM {{ source('ecommerce', 'customers') }}

{% endsnapshot %}""",
        tags=["snapshot", "scd", "check_strategy", "config"],
    ),
    DBTTemplate(
        tier="dbt",
        pattern="incremental",
        question="Write an incremental dbt model for subscription MRR history that only processes records updated in the last 7 days.",
        sql="""{{
  config(
    materialized       = 'incremental',
    unique_key         = 'subscription_id',
    incremental_strategy = 'merge',
    partition_by       = {'field': 'started_at', 'data_type': 'timestamp', 'granularity': 'month'},
    tags               = ['billing', 'mrr', 'incremental']
  )
}}

SELECT
  subscription_id,
  customer_id,
  plan_id,
  plan_name,
  status,
  started_at,
  cancelled_at,
  trial_ends_at,
  mrr_usd,
  billing_interval,
  CURRENT_TIMESTAMP() AS _dbt_updated_at
FROM {{ source('billing', 'subscriptions') }}

{% if is_incremental() %}
  WHERE started_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
     OR cancelled_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
{% endif %}""",
        tags=["incremental", "merge", "partition_by", "mrr"],
    ),
]


def get_dbt_templates_by_pattern(pattern: str) -> list[DBTTemplate]:
    return [t for t in DBT_TEMPLATES if t.pattern == pattern]


def get_dbt_templates_by_tag(tag: str) -> list[DBTTemplate]:
    return [t for t in DBT_TEMPLATES if tag in t.tags]
