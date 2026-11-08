import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "masanumasika_paricharya.json")
INDEX_OUTPUT_PATH = os.path.join(BASE_DIR, "ayurmitra_faiss.index")
METADATA_OUTPUT_PATH = os.path.join(BASE_DIR, "chunks_metadata.json")

def create_chunks_from_corpus(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    chunks = []
    
    for item in corpus.get("data", []):
        month = item["month"]
        month_name = item["sanskrit_name"]
        fetal_state = item["garbha_avastha"]
        modern_note = item["practical_guidelines"]["modern_correlation"]
        dos = ", ".join(item["practical_guidelines"]["pathya_do"])
        donts = ", ".join(item["practical_guidelines"]["apathya_dont"])
        
        # A. Classical Authors Chunks
        for author_key, details in item["classical_protocols"].items():
            author_text = (
                f"Month: {month} ({month_name}). Fetal Stage: {fetal_state}. "
                f"Authority: {details['grantha']} ({details['shloka_ref']}). "
                f"Sanskrit Shloka: {details['sanskrit_shloka']}. "
                f"Classical Regimen: {details['regimen']} "
                f"Clinical Rationale: {details['clinical_rationale']}"
            )
            
            chunks.append({
                "content": author_text,
                "metadata": {
                    "month": month,
                    "month_name": month_name,
                    "type": "classical_protocol",
                    "authority": details["grantha"],
                    "reference": details["shloka_ref"],
                    "sanskrit": details["sanskrit_shloka"],
                    "regimen": details["regimen"],
                    "rationale": details["clinical_rationale"]
                }
            })
        
        # B. Practical Daily Life & Modern Obstetric Correlation Chunk
        practical_text = (
            f"Month: {month} ({month_name}) Practical Diet & Lifestyle Guidelines. "
            f"Fetal Development Stage: {fetal_state}. "
            f"Pathya (Recommended Do's): {dos}. "
            f"Apathya (Contraindicated Don'ts): {donts}. "
            f"Modern Obstetric Correlation: {modern_note}"
        )
        
        chunks.append({
            "content": practical_text,
            "metadata": {
                "month": month,
                "month_name": month_name,
                "type": "practical_guidelines",
                "authority": "General Ayurvedic Consensus & Modern Obstetrics",
                "dos": item["practical_guidelines"]["pathya_do"],
                "donts": item["practical_guidelines"]["apathya_dont"],
                "modern_correlation": modern_note
            }
        })
        
    return chunks

def build_faiss_index():
    print("⏳ Reading Ayurvedic Corpus...")
    chunks = create_chunks_from_corpus(DATA_FILE)
    print(f"✅ Generated {len(chunks)} structured context chunks.")
    
    print("⏳ Loading Embedding Model (all-MiniLM-L6-v2)...")
    # Compact, fast 384-dimensional embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [c["content"] for c in chunks]
    print("⏳ Generating vector embeddings...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    dimension = embeddings.shape[1]
    
    print("⏳ Initializing FAISS Index (Cosine Similarity / Inner Product)...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"⏳ Saving FAISS index to {INDEX_OUTPUT_PATH}...")
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    
    print(f"⏳ Saving metadata to {METADATA_OUTPUT_PATH}...")
    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print("\n🎉 SUCCESS: Vector Index and Metadata successfully created!")
    print(f"Total vectors stored: {index.ntotal}")

if __name__ == "__main__":
    build_faiss_index()