"""
Streamlit playground for sql-forge.
Lets users paste a schema + question and get BigQuery/dbt SQL in real time.
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

EXAMPLE_SCHEMA = """table: orders
columns:
  - name: order_id     type: STRING   description: unique order identifier
  - name: customer_id  type: STRING   description: FK to customers
  - name: created_at   type: TIMESTAMP
  - name: status       type: STRING   values: [pending, shipped, delivered, cancelled]
  - name: amount_usd   type: FLOAT64"""

EXAMPLE_QUESTION = "For each customer, show their total spend and most recent order date. Only include customers with more than 3 orders."

st.set_page_config(page_title="sql-forge", page_icon="🔥", layout="wide")

st.title("🔥 sql-forge")
st.caption("BigQuery + dbt SQL generator — fine-tuned Phi-3-mini")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")

    dialect = st.radio("Dialect", ["bigquery", "dbt"], horizontal=True)

    schema_input = st.text_area(
        "Schema (YAML format)",
        value=EXAMPLE_SCHEMA,
        height=250,
    )

    question_input = st.text_area(
        "Question",
        value=EXAMPLE_QUESTION,
        height=80,
    )

    with st.expander("Advanced options"):
        max_tokens  = st.slider("Max new tokens", 64, 1024, 512)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)

    generate_btn = st.button("Generate SQL", type="primary", use_container_width=True)

with col2:
    st.subheader("Generated SQL")

    if generate_btn:
        if not schema_input.strip() or not question_input.strip():
            st.error("Please provide both schema and question.")
        else:
            with st.spinner("Generating..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/generate",
                        json={
                            "schema":          schema_input,
                            "question":        question_input,
                            "dialect":         dialect,
                            "max_new_tokens":  max_tokens,
                            "temperature":     temperature,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.code(data["sql"], language="sql")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Latency", f"{data['latency_ms']} ms")
                    m2.metric("Tokens",  data["tokens_used"])
                    m3.metric("Model",   data["model"])

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to API server. "
                        "Start it first with: `python inference/api.py`"
                    )
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.text}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
    else:
        st.info("Fill in the schema and question, then click **Generate SQL**.")

st.divider()
st.caption(
    "Model: sql-forge-phi3-v1 · "
    "Trained on BigQuery + dbt Jinja · "
    "Published at [Mpratham/sql-forge-phi3-v1](https://huggingface.co/Mpratham/sql-forge-phi3-v1)"
)
