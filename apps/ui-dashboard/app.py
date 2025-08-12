from __future__ import annotations

import os
import json
import requests
import streamlit as st

st.set_page_config(page_title="Consensus-DPO", page_icon="⚖️", layout="wide")

# Minimal, clean black/white theme via Streamlit config overrides
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #111111; }
    .stButton>button { background-color: #111111; color: #ffffff; border-radius: 6px; padding: 0.5rem 1rem; }
    .stTextInput>div>div>input { color: #111; }
    .css-ocqkz7, .e1f1d6gn5 { color: #111 !important; }
    .stMarkdown code { background: #f7f7f7; color: #111; }
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


