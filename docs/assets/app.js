/* ============================================================
   SQL Forge — Portfolio Landing Page JS
   ============================================================ */

// ── TypeWriter ────────────────────────────────────────────────
class TypeWriter {
  constructor(el, speed = 38) {
    this.el = el;
    this.speed = speed;
  }
  async type(text) {
    for (const ch of text) {
      this.el.textContent += ch;
      await wait(this.speed + (Math.random() * 20 - 10));
    }
  }
  async erase(keepChars = 0) {
    while (this.el.textContent.length > keepChars) {
      this.el.textContent = this.el.textContent.slice(0, -1);
      await wait(18);
    }
  }
  clear() { this.el.textContent = ''; }
}

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Hero terminal cycles ──────────────────────────────────────
const DEMOS = [
  {
    question: 'Show customer retention cohorts by signup month',
    sql: `SELECT
  FORMAT_DATE('%Y-%m', created_at) AS cohort,
  DATE_DIFF(act_month, cohort_date, MONTH) AS months_since,
  COUNT(DISTINCT user_id) AS retained_users
FROM (
  SELECT u.user_id,
    DATE_TRUNC(u.created_at, MONTH) AS cohort_date,
    DATE_TRUNC(e.event_time, MONTH) AS act_month
  FROM \`proj.ecommerce.users\` u
  JOIN \`proj.ecommerce.events\` e USING (user_id)
)
GROUP BY 1, 2
QUALIFY months_since <= 12
ORDER BY cohort, months_since`,
    lang: 'sql'
  },
  {
    question: 'Incremental dbt model — daily revenue by channel',
    sql: `{{ config(
  materialized = 'incremental',
  unique_key   = 'date_day'
) }}

WITH src AS (
  SELECT
    DATE_TRUNC(created_at, DAY)   AS date_day,
    acquisition_channel,
    SUM(amount_usd)               AS revenue_usd,
    COUNT(DISTINCT order_id)      AS orders
  FROM {{ ref('stg_orders') }}
  WHERE status != 'cancelled'
  {% if is_incremental() %}
    AND created_at > (
      SELECT MAX(date_day) FROM {{ this }}
    )
  {% endif %}
  GROUP BY 1, 2
)
SELECT * FROM src`,
    lang: 'sql'
  },
  {
    question: 'For each customer return only their latest order',
    sql: `SELECT
  order_id,
  customer_id,
  created_at,
  amount_usd,
  status
FROM \`proj.ecommerce.orders\`
WHERE status NOT IN ('cancelled','refunded')
QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY customer_id
    ORDER BY created_at DESC
  ) = 1`,
    lang: 'sql'
  }
];

let currentDemo = 0;
let heroRunning = false;

async function runHeroDemo(idx) {
  if (heroRunning) return;
  heroRunning = true;

  const demo = DEMOS[idx];
  const qEl   = document.getElementById('hero-question');
  const sqlEl = document.getElementById('hero-sql');
  const pre   = document.getElementById('hero-pre');
  const dots  = document.querySelectorAll('.cycle-dot');

  dots.forEach((d, i) => d.classList.toggle('active', i === idx));

  const qTW = new TypeWriter(qEl);
  await qTW.erase(0);
  await wait(200);
  await qTW.type(demo.question);
  await wait(500);

  pre.style.opacity = '0';
  await wait(200);
  sqlEl.textContent = demo.sql;
  if (window.Prism) Prism.highlightElement(sqlEl);
  pre.style.opacity = '1';
  pre.style.transition = 'opacity 0.4s';

  heroRunning = false;
}

// ── Scroll fade-in ────────────────────────────────────────────
function initScrollObserver() {
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll('.fade-in').forEach(el => io.observe(el));
}

// ── Code tabs ─────────────────────────────────────────────────
function initCodeTabs() {
  document.querySelectorAll('.code-tabs').forEach(tabs => {
    tabs.querySelectorAll('.code-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const key = tab.dataset.tab;
        const wrapper = tabs.closest('.code-showcase');
        wrapper.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
        wrapper.querySelectorAll('.code-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        wrapper.querySelector(`.code-panel[data-panel="${key}"]`).classList.add('active');
        if (window.Prism) Prism.highlightAllUnder(wrapper);
      });
    });
  });
}

// ── Pipeline step highlight ───────────────────────────────────
function initPipeline() {
  document.querySelectorAll('.pipeline-step').forEach(step => {
    step.addEventListener('mouseenter', () => {
      document.querySelectorAll('.pipeline-step').forEach(s => s.classList.remove('active'));
      step.classList.add('active');
    });
  });
}

// ── Mock demo generation ──────────────────────────────────────
const MOCK_OUTPUTS = {
  cohort: {
    sql: `SELECT
  FORMAT_DATE('%Y-%m', u.created_at) AS cohort_month,
  DATE_DIFF(
    DATE_TRUNC(e.event_time, MONTH),
    DATE_TRUNC(u.created_at, MONTH),
    MONTH
  ) AS months_since_signup,
  COUNT(DISTINCT u.user_id) AS retained_users,
  COUNT(DISTINCT u.user_id) / FIRST_VALUE(COUNT(DISTINCT u.user_id))
    OVER (PARTITION BY FORMAT_DATE('%Y-%m', u.created_at)
          ORDER BY DATE_DIFF(DATE_TRUNC(e.event_time,MONTH),
                             DATE_TRUNC(u.created_at,MONTH),MONTH)
          ROWS UNBOUNDED PRECEDING) AS retention_rate
FROM \`project.ecommerce.users\` u
JOIN \`project.ecommerce.events\` e USING (user_id)
GROUP BY 1, 2
QUALIFY months_since_signup BETWEEN 0 AND 12
ORDER BY cohort_month, months_since_signup`,
    meta: { parse: '✓ Valid BigQuery', tokens: 187, latency: '142ms', features: 'QUALIFY · FORMAT_DATE · window fn' }
  },
  incremental: {
    sql: `{{ config(
  materialized = 'incremental',
  unique_key   = 'date_day',
  tags         = ['revenue', 'daily']
) }}

WITH orders AS (
  SELECT
    DATE_TRUNC(created_at, DAY) AS date_day,
    acquisition_channel,
    SUM(amount_usd)             AS revenue_usd,
    COUNT(DISTINCT order_id)    AS order_count,
    COUNT(DISTINCT customer_id) AS unique_customers
  FROM {{ ref('stg_orders') }}
  WHERE status != 'cancelled'
  {% if is_incremental() %}
    AND created_at >= (
      SELECT TIMESTAMP_SUB(MAX(date_day), INTERVAL 3 DAY)
      FROM {{ this }}
    )
  {% endif %}
  GROUP BY 1, 2
)
SELECT * FROM orders`,
    meta: { parse: '✓ Valid dbt + BQ', tokens: 156, latency: '118ms', features: 'is_incremental() · ref() · config()' }
  },
  qualify: {
    sql: `SELECT
  order_id,
  customer_id,
  created_at,
  amount_usd,
  status,
  LAG(amount_usd) OVER (
    PARTITION BY customer_id
    ORDER BY created_at
  ) AS prev_order_amount
FROM \`project.ecommerce.orders\`
WHERE status NOT IN ('cancelled', 'refunded')
QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY customer_id
    ORDER BY created_at DESC
  ) = 1`,
    meta: { parse: '✓ Valid BigQuery', tokens: 121, latency: '98ms', features: 'QUALIFY · ROW_NUMBER() · LAG()' }
  },
  default: {
    sql: `SELECT
  DATE_TRUNC(o.created_at, MONTH) AS month,
  c.acquisition_channel,
  COUNT(DISTINCT o.customer_id)   AS unique_customers,
  SUM(o.amount_usd)               AS total_revenue_usd,
  AVG(o.amount_usd)               AS avg_order_value,
  PERCENTILE_CONT(o.amount_usd, 0.5) OVER (
    PARTITION BY DATE_TRUNC(o.created_at, MONTH)
  ) AS median_order_value
FROM \`project.ecommerce.orders\` o
JOIN \`project.ecommerce.customers\` c
  ON o.customer_id = c.customer_id
WHERE o.status = 'delivered'
  AND o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
GROUP BY 1, 2
ORDER BY month DESC, total_revenue_usd DESC`,
    meta: { parse: '✓ Valid BigQuery', tokens: 174, latency: '134ms', features: 'PERCENTILE_CONT · DATE_TRUNC · TIMESTAMP_SUB' }
  }
};

function pickOutput(question, dialect) {
  const q = question.toLowerCase();
  if (dialect === 'dbt' || q.includes('incremental') || q.includes('dbt') || q.includes('model')) {
    return MOCK_OUTPUTS.incremental;
  }
  if (q.includes('cohort') || q.includes('retention')) return MOCK_OUTPUTS.cohort;
  if (q.includes('latest') || q.includes('recent') || q.includes('last order') || q.includes('qualify')) {
    return MOCK_OUTPUTS.qualify;
  }
  return MOCK_OUTPUTS.default;
}

function initDemo() {
  const btn = document.getElementById('demo-generate-btn');
  const questionEl = document.getElementById('demo-question');
  const outputEl = document.getElementById('demo-sql-output');
  const spinnerEl = document.getElementById('demo-spinner');
  const metaEl = document.getElementById('demo-meta');
  const dialectBtns = document.querySelectorAll('.dialect-btn');

  let selectedDialect = 'bigquery';
  dialectBtns.forEach(b => {
    b.addEventListener('click', () => {
      dialectBtns.forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      selectedDialect = b.dataset.dialect;
    });
  });

  btn.addEventListener('click', async () => {
    const question = questionEl.value.trim();
    if (!question) {
      questionEl.focus();
      return;
    }
    btn.disabled = true;
    spinnerEl.classList.add('active');
    metaEl.innerHTML = '';

    await wait(900 + Math.random() * 400);

    const out = pickOutput(question, selectedDialect);
    spinnerEl.classList.remove('active');

    outputEl.textContent = out.sql;
    outputEl.className = 'language-sql';
    if (window.Prism) Prism.highlightElement(outputEl);

    metaEl.innerHTML = `
      <span class="meta-chip pass">🔍 Parse: ${out.meta.parse}</span>
      <span class="meta-chip info">⚡ ${out.meta.latency}</span>
      <span class="meta-chip info">🪙 ${out.meta.tokens} tokens</span>
      <span class="meta-chip info">✨ ${out.meta.features}</span>
    `;
    btn.disabled = false;
  });

  // Pre-fill example
  const exampleBtn = document.getElementById('demo-example-btn');
  if (exampleBtn) {
    exampleBtn.addEventListener('click', () => {
      questionEl.value = 'Show monthly revenue trends by acquisition channel for the last 90 days';
      questionEl.focus();
    });
  }
}

// ── Cycle dots ────────────────────────────────────────────────
function initCycleDots() {
  document.querySelectorAll('.cycle-dot').forEach((dot, idx) => {
    dot.addEventListener('click', () => {
      currentDemo = idx;
      runHeroDemo(idx);
    });
  });
}

// ── Smooth nav + active link ──────────────────────────────────
function initNav() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const id = a.getAttribute('href').slice(1);
      const el = document.getElementById(id);
      if (el) { e.preventDefault(); el.scrollIntoView({ behavior: 'smooth' }); }
    });
  });
}

// ── Auto-cycle hero every 8 seconds ──────────────────────────
function startHeroCycle() {
  setInterval(() => {
    if (!heroRunning) {
      currentDemo = (currentDemo + 1) % DEMOS.length;
      runHeroDemo(currentDemo);
    }
  }, 8000);
}

// ── Highlight all code blocks once Prism is ready ────────────
function highlightAll() {
  if (window.Prism) Prism.highlightAll();
}

// ── Init ─────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  initScrollObserver();
  initCodeTabs();
  initPipeline();
  initDemo();
  initCycleDots();
  initNav();

  // Start hero after a short delay so page renders first
  wait(400).then(() => {
    runHeroDemo(0);
    startHeroCycle();
  });

  // Highlight static code blocks
  wait(600).then(highlightAll);
});
