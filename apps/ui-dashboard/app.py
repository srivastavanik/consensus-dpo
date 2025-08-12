from __future__ import annotations

import os
import json
import requests
import streamlit as st

st.set_page_config(page_title="Consensus-DPO", page_icon="⚖️", layout="wide")

# Minimal, clean black/white theme via Streamlit config overrides
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    :root { --primary: #000000; --text: #111111; --bg: #ffffff; }
    html, body, .stApp { background-color: var(--bg); color: var(--text); font-family: 'Inter', -apple-system, system-ui, 'Segoe UI', Roboto, Helvetica, Arial, 'Noto Sans', sans-serif; }
    h1, h2, h3, h4 { color: var(--text); font-weight: 600; }
    .stButton>button { background-color: var(--primary); color: #ffffff; border-radius: 6px; padding: 0.55rem 1rem; border: 1px solid var(--primary); }
    .stButton>button:hover { opacity: 0.9; }
    textarea, input, select { background-color: #ffffff !important; color: var(--text) !important; border: 1px solid var(--primary) !important; border-radius: 6px !important; }
    .stTextArea textarea, .stTextInput input, .stNumberInput input { background: #ffffff !important; color: var(--text) !important; }
    .stNumberInput > div, .stTextInput > div, .stTextArea > div { background: #ffffff !important; }
    ::placeholder { color: #666666 !important; opacity: 1; }
    .stTabs [data-baseweb="tab"] { color: #666; }
    .stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; }
    .stMarkdown code { background: #f2f2f2; color: var(--text); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Consensus → DPO")

API = os.getenv("ORCH_URL", "http://127.0.0.1:8000")

with st.form("consensus_form", clear_on_submit=False):
    prompt = st.text_area("Prompt", height=140, placeholder="Ask a question or give a task…")
    model = st.text_input("Model", value="gpt-oss-small")
    k = st.number_input("K candidates", min_value=2, max_value=7, value=3, step=1)
    m = st.number_input("m judge views", min_value=1, max_value=4, value=2, step=1)
    r = st.number_input("R debate rounds", min_value=0, max_value=2, value=1, step=1)
    submitted = st.form_submit_button("Run consensus")

tab_run, tab_eval = st.tabs(["Run", "Evaluate"])

with tab_run:
    if submitted and prompt.strip():
        try:
            resp = requests.post(
                f"{API}/consensus",
                json={
                    "prompt": prompt,
                    "model": model,
                    "k": int(k),
                    "m": int(m),
                    "r": int(r),
                },
            )
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} {resp.text}")
            else:
                data = resp.json()
                st.subheader("Final decision")
                st.json(data.get("final", {}))
                st.subheader("All views")
                st.json(data.get("decisions", []))
                st.caption(f"Pair written: {data.get('pair_written')} → {data.get('pairs_path')}")
                if data.get("evidence_supported") is not None:
                    st.caption(f"Evidence supported: {data['evidence_supported']}")
        except Exception as e:
            st.error(str(e))

with tab_eval:
    st.write("GSM8K quick evaluation (EM on limited subset)")
    n = st.number_input("Limit", min_value=10, max_value=200, value=20, step=10)
    if st.button("Run GSM8K eval"):
        try:
            import subprocess, sys, json as _json
            env = os.environ.copy()
            env["ORCH_URL"] = API
            out = subprocess.check_output([sys.executable, "apps/evaluator/bench_gsm8k.py"], env=env)
            st.json(_json.loads(out.decode().strip().replace("'", '"')) if out else {"msg": "done"})
        except Exception as e:
            st.error(str(e))
    st.write("MMLU quick evaluation (subset)")
    n2 = st.number_input("Limit (MMLU)", min_value=20, max_value=200, value=50, step=10)
    if st.button("Run MMLU eval"):
        try:
            import subprocess, sys, json as _json
            env = os.environ.copy()
            env["ORCH_URL"] = API
            out = subprocess.check_output([sys.executable, "apps/evaluator/bench_mmlu.py"], env=env)
            st.json(_json.loads(out.decode().strip().replace("'", '"')) if out else {"msg": "done"})
        except Exception as e:
            st.error(str(e))


