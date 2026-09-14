"""
Simple Streamlit test UI for the FasalGuru FastAPI backend.

This is ONLY a testing tool — it does not replace or modify your FastAPI
backend. It just sends requests to it and shows the response nicely.

HOW TO RUN:
1. Make sure your FastAPI server is already running in another terminal:
     uvicorn app.main:app --reload --port 8000
2. In a SECOND terminal (same .venv activated), run:
     streamlit run streamlit_test_app.py
3. A browser tab will open automatically at http://localhost:8501
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/krishi-agent/ask"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(page_title="FasalGuru - Krishi Agent Test", page_icon="🌾")
st.title("🌾 FasalGuru — Krishi Agent (Test UI)")
st.caption("Ye sirf testing ke liye hai. Ye FastAPI backend (port 8000) ko call karta hai.")

# --- Backend health check ---
with st.sidebar:
    st.subheader("Backend status")
    try:
        health = requests.get(HEALTH_URL, timeout=5).json()
        if health.get("vector_store_loaded"):
            st.success("✅ Backend chal raha hai, vector store loaded hai.")
        else:
            st.warning("⚠️ Backend chal raha hai, lekin vector store load nahi hua. "
                        "Pehle ingestion chalao: python -m app.ingestion.ingest_docs")
    except requests.exceptions.RequestException:
        st.error("❌ Backend se connect nahi ho pa raha. Kya uvicorn chal raha hai "
                  "(uvicorn app.main:app --reload --port 8000)?")

# --- Query form ---
with st.form("ask_form"):
    query = st.text_area("Aapka sawaal (Hindi ya English mein)", height=100,
                          placeholder="e.g. Chane me pili patti kyon ho rahi hai?")
    col1, col2 = st.columns(2)
    with col1:
        crop = st.text_input("Fasal (optional)", placeholder="e.g. chana")
    with col2:
        district = st.text_input("Zila (optional)", placeholder="e.g. sitapur")
    submitted = st.form_submit_button("Poocho 🌱")

if submitted:
    if not query.strip():
        st.error("Pehle apna sawaal likho.")
    else:
        payload = {"query": query.strip()}
        if crop.strip():
            payload["crop"] = crop.strip()
        if district.strip():
            payload["district"] = district.strip()

        with st.spinner("Sochte hue... 🤔"):
            try:
                resp = requests.post(API_URL, json=payload, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.subheader("Jawab:")
                    st.write(data["answer"])

                    st.caption(f"Provider used: `{data.get('provider_used', 'unknown')}`")

                    sources = data.get("sources", [])
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)})"):
                            for i, s in enumerate(sources, 1):
                                st.markdown(f"**{i}.** {s}")
                else:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request fail ho gayi: {e}")