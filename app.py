import logging
import os
from pathlib import Path
from urllib.request import urlopen
import streamlit as st

from config import CATALOGUE, POLICY
from trustcart.core import ProductIndex, answer_query, load_catalogue, load_policy

logging.basicConfig(level=logging.INFO)


def ensure_runtime_data():
    """Use local challenge files when present; otherwise fetch from a trusted HTTPS host."""
    from config import CATALOGUE, POLICY

    if CATALOGUE.is_file() and POLICY.is_file():
        return

    base_url = ""
    try:
        base_url = st.secrets.get("DATA_BASE_URL", "")
    except Exception:
        base_url = os.environ.get("DATA_BASE_URL", "")

    if not base_url:
        return

    if not base_url.startswith("https://"):
        raise ValueError("DATA_BASE_URL must use HTTPS.")

    CATALOGUE.parent.mkdir(parents=True, exist_ok=True)
    for filename, destination in (
        ("catalogue.csv", CATALOGUE),
        ("policy.md", POLICY),
    ):
        if destination.is_file():
            continue
        url = base_url.rstrip("/") + "/" + filename
        with urlopen(url, timeout=20) as response:
            content = response.read()
        if not content:
            raise ValueError(f"The hosted data file {filename} is empty.")
        destination.write_bytes(content)


st.set_page_config(page_title="TrustCart AI", page_icon="🛒", layout="wide")
st.markdown("""
<style>
.stApp {background:#f5f7fb}
.block-container {max-width:1100px;padding-top:2rem}
.hero {padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#10243a,#176b68);color:white}
.hero h1 {color:white;margin:0}
.hero p {color:#e5f4f2;margin:.35rem 0 0}
</style>
<div class="hero"><h1>🛒 TrustCart AI</h1>
<p>The Conversational Shopping Assistant That Doesn't Lie · R3</p></div>
""", unsafe_allow_html=True)

ensure_runtime_data()

@st.cache_resource(show_spinner="Validating official catalogue and policy…")
def resources():
    products = load_catalogue(CATALOGUE)
    policy = load_policy(POLICY)
    return products, ProductIndex(products), policy

with st.sidebar:
    st.subheader("Trust controls")
    st.success("Catalogue-grounded answers")
    st.info("Rules + retrieval · no LLM calls")
    st.caption("Synthetic catalogue snapshot: 1 October 2026")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

try:
    products, index, policy = resources()
except (FileNotFoundError, ValueError) as exc:
    st.error("Official R3 data is not configured.")
    st.code(str(exc))
    st.info("Copy catalogue.csv and policy.md from the challenge pack into FULL_CODE/data/.")
    st.stop()

st.caption(f"Validated records: {len(products):,} products · {len(policy)} policy subsections")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Ask about products, price, stock or store policy…", max_chars=500)
if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking catalogue evidence…"):
            answer = answer_query(prompt, st.session_state.history, products, index, policy)
        st.markdown(answer.answer_text)
        label = {"answer": "ANSWER", "clarify": "CLARIFY", "refuse": "REFUSE"}[answer.action]
        meta = f"{label} · {answer.latency_ms:.2f} ms · tokens in/out {answer.tokens_in}/{answer.tokens_out}"
        st.caption(meta)
        with st.expander("Evidence and structured response"):
            st.json(answer.to_dict())
    st.session_state.history = (st.session_state.history + [prompt])[-8:]
    st.session_state.messages.append({"role": "assistant", "text": answer.answer_text, "meta": meta})

if not st.session_state.messages:
    st.markdown("### Try these questions")
    st.write("- `65W charger dikhao`")
    st.write("- `What is the return window?`")
    st.write("- `Do you sell iPhones?`")
    st.write("- `Sanchayo NimbusMax steel container set stock me hai?`")
