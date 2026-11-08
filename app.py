import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="AyurMitra Backend API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "masanumasika_paricharya.json")

corpus = []
vectorizer = None
tfidf_matrix = None

@app.on_event("startup")
def load_data_and_index():
    global corpus, vectorizer, tfidf_matrix
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Corpus file not found at {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = []
    documents = []

    for month_entry in data.get("months", []):
        m_num = month_entry.get("month")
        m_name = month_entry.get("month_name")
        modern = month_entry.get("modern_obstetric_correlation", "")
        dos = month_entry.get("general_pathya_dos", [])
        donts = month_entry.get("general_apathya_donts", [])

        for auth in month_entry.get("authorities", []):
            item = {
                "month": m_num,
                "month_name": m_name,
                "authority": auth.get("authority", ""),
                "reference": auth.get("reference", ""),
                "sanskrit": auth.get("sanskrit", ""),
                "regimen": auth.get("regimen", ""),
                "rationale": auth.get("rationale", ""),
                "dos": dos,
                "donts": donts,
                "modern_correlation": modern
            }
            corpus.append(item)
            doc_text = f"Month {m_num} {m_name} {item['authority']} {item['reference']} {item['regimen']} {item['rationale']} {item['sanskrit']} {' '.join(dos)} {' '.join(donts)} {modern}"
            documents.append(doc_text)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

@app.get("/")
def home():
    return {
        "project": "AyurMitra",
        "tagline": "Grounded in Classical Ayurvedic Literature",
        "status": "Online",
        "engine": "Optimized Vector Retrieval",
        "indexed_chunks": len(corpus)
    }

@app.post("/api/retrieve")
def retrieve_contexts(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_vec = vectorizer.transform([req.query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:req.top_k]

    results = []
    for idx in top_indices:
        item = corpus[idx].copy()
        item["similarity_score"] = round(float(scores[idx]), 4)
        results.append(item)

    return {
        "query": req.query,
        "inference": {
            "status": "standby",
            "model": "AyurParam 2.9B (BharatGen)",
            "message": "Serving verified classical citations directly from Brihat-trayi corpus."
        },
        "results_count": len(results),
        "citations": results,
        "disclaimer": "Derived from authentic Ayurvedic Samhitas for academic and clinical reference."
    }
