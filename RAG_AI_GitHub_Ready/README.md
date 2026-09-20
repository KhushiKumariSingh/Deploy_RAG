# RAG AI Teaching Assistant

A Streamlit RAG application that retrieves relevant subtitle chunks from the Sigma Web Development course and uses a hosted LLM to answer with the relevant video and timestamp.

## Deployment

1. Upload the contents of this folder to a GitHub repository.
2. Deploy `app.py` from Streamlit Community Cloud.
3. In Streamlit Cloud, add these secrets:

```toml
HF_TOKEN = "..."
GROQ_API_KEY = "..."
```

The app uses Hugging Face Inference for BAAI/bge-m3 query embeddings and Groq for answer generation. The precomputed `embeddings.joblib` file must remain in the repository.

## Main file

`app.py`

## Important

Do not upload `.venv`, `.git`, or API keys.
