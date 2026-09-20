import streamlit as st
from rag_engine import answer_question

st.set_page_config(page_title="RAG AI Teaching Assistant", page_icon="🎓", layout="centered")

st.title("🎓 RAG AI Teaching Assistant")
st.caption("Ask questions about the Sigma Web Development course and get the relevant video and timestamp.")

question = st.text_input("Ask a question", placeholder="e.g. Where are CSS selectors explained?")

if st.button("Submit", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the course and generating an answer..."):
            try:
                result = answer_question(question.strip())
                st.markdown(result)
            except Exception as exc:
                st.error(f"The assistant could not answer right now: {exc}")
                st.info("Check that HF_TOKEN and GROQ_API_KEY are configured in Streamlit Secrets.")
