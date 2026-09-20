import os
from pathlib import Path

import joblib
import numpy as np
from groq import Groq
from huggingface_hub import InferenceClient

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = BASE_DIR / "embeddings.joblib"
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-m3")
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_RESULTS = 5


def _get_secret(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    raise RuntimeError(f"Missing {name}. Add it to Streamlit Secrets or environment variables.")


@__import__("functools").lru_cache(maxsize=1)
def load_embeddings():
    return joblib.load(EMBEDDINGS_PATH)


@__import__("functools").lru_cache(maxsize=1)
def get_hf_client():
    return InferenceClient(provider="hf-inference", api_key=_get_secret("HF_TOKEN"))


@__import__("functools").lru_cache(maxsize=1)
def get_groq_client():
    return Groq(api_key=_get_secret("GROQ_API_KEY"))


def create_embedding(text: str) -> np.ndarray:
    result = get_hf_client().feature_extraction(text, model=EMBEDDING_MODEL)
    embedding = np.asarray(result, dtype=np.float32)
    if embedding.ndim == 2:
        embedding = embedding[0]
    if embedding.ndim != 1:
        raise RuntimeError(f"Unexpected embedding shape: {embedding.shape}")
    return embedding


def retrieve(question: str, top_k: int = TOP_RESULTS):
    df = load_embeddings()
    query_embedding = create_embedding(question)

    matrix = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError("The stored embeddings are not in the expected 2-D format.")
    if matrix.shape[1] != query_embedding.shape[0]:
        raise RuntimeError(
            f"Embedding dimension mismatch: stored vectors have {matrix.shape[1]} dimensions, "
            f"but the query embedding has {query_embedding.shape[0]}."
        )

    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    similarities = (matrix @ query_embedding) / (matrix_norms * query_norm + 1e-12)

    indices = np.argsort(similarities)[::-1][:top_k]
    return df.iloc[indices].copy(), similarities[indices]


def build_prompt(question: str, results) -> str:
    context = results[["title", "number", "start", "end", "text"]].to_json(orient="records")

    return f"""You are the teaching assistant for a Sigma Web Development course.

Use ONLY the supplied course subtitle chunks to answer the user's question.

Course chunks:
{context}

User question:
{question}

Instructions:
- Answer clearly and naturally.
- If the question is related to the course, tell the learner which video covers the topic and the approximate timestamp (start time in seconds converted to minutes/seconds when useful).
- Give a short explanation based on the retrieved chunks.
- If the question is unrelated to the course, say that you can only answer questions related to this course.
- Do not mention embeddings, retrieval, chunks, JSON, prompts, or internal implementation details.
"""


def generate_answer(prompt: str) -> str:
    completion = get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a concise, helpful course teaching assistant. Follow the supplied course context strictly.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=700,
    )
    return completion.choices[0].message.content.strip()


def answer_question(question: str) -> str:
    results, _ = retrieve(question)
    prompt = build_prompt(question, results)
    return generate_answer(prompt)
