import os
import json
import requests
import streamlit as st

# Clean, minimal config
st.set_page_config(
    page_title="Consensus → DPO",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple CSS
st.markdown("""
<style>
    .stApp {
        background-color: white;
        color: black;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stButton > button {
        background-color: black;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
    }
    .stTextInput, .stTextArea, .stNumberInput {
        color: black;
    }
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        color: black !important;
        background: white !important;
        border: 1px solid #ccc !important;
    }
    .stMarkdown {
        color: black;
    }
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Consensus → DPO")

API_URL = os.getenv("ORCH_URL", "http://127.0.0.1:8000")

# Main form
with st.form("consensus_form"):
    prompt = st.text_area("Prompt", height=120, placeholder="Enter your question or task...")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model = st.text_input("Model", value="gpt-oss-small")
    with col2:
        k = st.number_input("K candidates", min_value=2, max_value=7, value=3)
    with col3:
        m = st.number_input("m judge views", min_value=1, max_value=4, value=2)
    with col4:
        r = st.number_input("R debate rounds", min_value=0, max_value=2, value=1)
    
    submitted = st.form_submit_button("Run Consensus", use_container_width=True)

# Results
if submitted and prompt.strip():
    with st.spinner("Running consensus pipeline..."):
        try:
            response = requests.post(
                f"{API_URL}/consensus",
                json={
                    "prompt": prompt,
                    "model": model,
                    "k": int(k),
                    "m": int(m),
                    "r": int(r)
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                
                st.success("✅ Consensus completed!")
                
                # Final decision
                st.subheader("Final Decision")
                final = data.get("final", {})
                st.json(final)
                
                # All judge views
                st.subheader("All Judge Views")
                decisions = data.get("decisions", [])
                for i, decision in enumerate(decisions):
                    st.write(f"**View {i+1}:**")
                    st.json(decision)
                
                # Metadata
                st.subheader("Metadata")
                st.write(f"Pair written: {data.get('pair_written', False)}")
                st.write(f"Evidence supported: {data.get('evidence_supported', 'N/A')}")
                st.write(f"Pairs file: {data.get('pairs_path', 'N/A')}")
                
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Evaluation section
st.divider()
st.subheader("Quick Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.write("**GSM8K Math**")
    gsm8k_limit = st.number_input("GSM8K Limit", min_value=5, max_value=100, value=20, key="gsm8k")
    if st.button("Run GSM8K", key="gsm8k_btn"):
        with st.spinner("Running GSM8K evaluation..."):
            try:
                # Simple direct call to avoid import issues
                import subprocess
                import sys
                env = os.environ.copy()
                env["ORCH_URL"] = API_URL
                result = subprocess.run([
                    sys.executable, "apps/evaluator/bench_gsm8k.py"
                ], env=env, capture_output=True, text=True, cwd=".")
                if result.returncode == 0:
                    st.success("GSM8K completed!")
                    st.text(result.stdout)
                else:
                    st.error(f"GSM8K failed: {result.stderr}")
            except Exception as e:
                st.error(f"GSM8K error: {str(e)}")

with col2:
    st.write("**MMLU Knowledge**")
    mmlu_limit = st.number_input("MMLU Limit", min_value=10, max_value=200, value=50, key="mmlu")
    if st.button("Run MMLU", key="mmlu_btn"):
        with st.spinner("Running MMLU evaluation..."):
            try:
                import subprocess
                import sys
                env = os.environ.copy()
                env["ORCH_URL"] = API_URL
                result = subprocess.run([
                    sys.executable, "apps/evaluator/bench_mmlu.py"
                ], env=env, capture_output=True, text=True, cwd=".")
                if result.returncode == 0:
                    st.success("MMLU completed!")
                    st.text(result.stdout)
                else:
                    st.error(f"MMLU failed: {result.stderr}")
            except Exception as e:
                st.error(f"MMLU error: {str(e)}")

# Footer
st.divider()
st.caption("Consensus → DPO: Multi-agent reasoning with preference learning")