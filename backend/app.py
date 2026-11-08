import os
import json
import urllib.request
import urllib.error
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="AyurMitra Backend API", version="1.1.0")

# Enable CORS for React frontend (Netlify / local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "ayurmitra_faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "chunks_metadata.json")

# Environment variables for AyurParam inference endpoint
AYURPARAM_API_URL = os.getenv("AYURPARAM_API_URL", "")
AYURPARAM_API_KEY = os.getenv("AYURPARAM_API_KEY", "")

# Global variables for loaded assets
index = None
metadata = []
embed_model = None

@app.on_event("startup")
def load_rag_pipeline():
    global index, metadata, embed_model
    print("⏳ Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("⏳ Loading FAISS index...")
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise RuntimeError("FAISS index or metadata not found. Run ingest.py first!")
        
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ RAG Pipeline Loaded with {len(metadata)} indexed contexts.")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

def call_ayurparam_inference(prompt: str) -> dict:
    """Calls the AyurParam inference endpoint or returns verified standby status."""
    if not AYURPARAM_API_URL:
        return {
            "status": "standby",
            "model": "AyurParam 2.9B (BharatGen)",
            "message": "Inference endpoint URL not configured. Operating in Direct-RAG Verified Evidence mode.",
            "synthesized_answer": None
        }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AYURPARAM_API_KEY}" if AYURPARAM_API_KEY else ""
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    try:
        req = urllib.request.Request(
            AYURPARAM_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=25) as response:
            result = json.loads(response.read().decode("utf-8"))
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                text = result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                text = result["generated_text"]
            else:
                text = str(result)
            return {
                "status": "online",
                "model": "AyurParam 2.9B (BharatGen)",
                "synthesized_answer": text.strip()
            }
    except Exception as e:
        return {
            "status": "offline_fallback",
            "model": "AyurParam 2.9B (BharatGen)",
            "message": f"AyurParam endpoint unavailable ({str(e)}). Serving authentic RAG evidence directly.",
            "synthesized_answer": None
        }

@app.get("/")
def home():
    return {
        "project": "AyurMitra",
        "tagline": "Grounded in Classical Ayurvedic Literature",
        "status": "Online",
        "inference_engine": "AyurParam 2.9B (BharatGen)",
        "indexed_chunks": len(metadata) if metadata else 0
    }

@app.post("/api/retrieve")
def retrieve_and_synthesize(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    query_vector = embed_model.encode([req.query], normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype=np.float32)
    
    k = min(req.top_k, index.ntotal)
    distances, indices = index.search(query_vector, k)
    
    results = []
    context_blocks = []
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            item = metadata[idx]
            cite_data = {
                "similarity_score": round(float(dist), 4),
                "month": item["metadata"]["month"],
                "month_name": item["metadata"]["month_name"],
                "authority": item["metadata"].get("authority", ""),
                "reference": item["metadata"].get("reference", ""),
                "sanskrit": item["metadata"].get("sanskrit", ""),
                "regimen": item["metadata"].get("regimen", ""),
                "rationale": item["metadata"].get("rationale", ""),
                "dos": item["metadata"].get("dos", []),
                "donts": item["metadata"].get("donts", []),
                "modern_correlation": item["metadata"].get("modern_correlation", "")
            }
            results.append(cite_data)
            
            # Format context block for AyurParam prompt
            context_blocks.append(
                f"[{cite_data['authority']} - {cite_data['reference']}] "
                f"Month: {cite_data['month']}. Regimen: {cite_data['regimen']}. "
                f"Sanskrit: {cite_data['sanskrit']}. Rationale: {cite_data['rationale']}"
            )
            
    # Assemble Grounded Prompt for AyurParam 2.9B
    context_str = "\n\n".join(context_blocks)
    prompt = (
        f"<|system|>\nYou are AyurMitra, a clinical knowledge assistant grounded in classical Ayurvedic texts. "
        f"Answer the query accurately using ONLY the provided classical evidence below. "
        f"Always cite the Samhita and shloka references (e.g. [Cha. Sha. 8/32]).\n\n"
        f"CONTEXT EVIDENCE:\n{context_str}\n\n"
        f"<|user|>\n{req.query}\n\n"
        f"<|assistant|>\n"
    )

    inference_result = call_ayurparam_inference(prompt)
            
    return {
        "query": req.query,
        "inference": inference_result,
        "results_count": len(results),
        "citations": results,
        "disclaimer": "This information is derived from authentic Ayurvedic Samhitas for academic and clinical reference only."
    }