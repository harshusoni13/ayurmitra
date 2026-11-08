# 🌿 AyurMitra (AyurGarbha Core)
### An AyurParam-Powered, Source-Grounded Ayurvedic Maternal Knowledge & Research Assistant

> **Developed for:** IIT Bombay Techfest — *AyushTech Innovate: National Open Ayurveda AI Challenge*  
> **Institutional Affiliation:** Govt. Autonomous Ashtang Ayurved Medical College & Hospital, Indore (M.P.)  
> **Licence:** MIT Permissive Open Source  

---

## 📌 Executive Summary

**AyurMitra** is a verifiable, retrieval-augmented intelligence assistant built specifically for Ayurveda scholars, BAMS students, clinicians, and researchers. Unlike generic LLMs that frequently hallucinate Sanskrit verses and fabricate page numbers, AyurMitra operates on a **strictly grounded RAG pipeline** indexed against authentic classical Ayurvedic treatises:

- **Charaka Samhita** (*Sharira Sthana, Adhyaya 8*)
- **Sushruta Samhita** (*Sharira Sthana, Adhyaya 10*)
- **Ashtanga Hridaya & Ashtanga Sangraha** (*Sharira Sthana, Adhyaya 1*)
- **Kashyapa Samhita** (*Sharira Sthana, Garbhini Chikitsita Adhyaya*)

Its initial flagship knowledge domain is **Masanumasika Garbhini Paricharya** (month-by-month maternal regimen from Month 1 to Month 9), providing side-by-side author comparisons, original Devanagari shlokas, clinical rationales, practical Pathya/Apathya checklists, and modern obstetric correlations.

---

## 🏛️ System Architecture

```text
[ USER QUERY ] (Hindi / English / Hinglish)
       │
       ▼
[ REACT RESPONSIVE FRONTEND ] (Tailwind CSS, Single-page, mobile-optimized)
       │
       ▼ (REST API: /api/retrieve)
[ FASTAPI BACKEND API GATEWAY ]
       │
       ▼
[ RETRIEVAL-AUGMENTED GENERATION (RAG) PIPELINE ]
  ├── 1. Sentence-Transformers Vector Embedding Model (all-MiniLM-L6-v2)
  ├── 2. Vector Database: FAISS (Flat Inner Product - Cosine Metric)
  └── 3. Metadata Chunker: Structured classical shlokas & clinical rationales
       │
       ▼
[ AYURPARAM 2.9B INFERENCE LAYER ] (BharatGen / IIT Bombay)
  ├── System Prompt: Strict source-grounding & citation enforcement
  ├── Retrieved Context Injection: Exact classical verses and commentaries
  └── Transparent Fail-safe: Direct verified RAG evidence if endpoint on standby
       │
       ▼
[ CITATION & VERIFICATION CARDS ] (Mula Shloka + Regimen + Rationale + Pathya/Apathya)
✨ Key Technical & Clinical Highlights
Zero Hallucination Protocol: System answers only using verified retrieved classical context. If evidence is not present in classical texts, it explicitly acknowledges limitations rather than generating speculative claims.

Traceable Citations: Every single output card displays Grantha name, Sthana, Adhyaya, Shloka number, and verbatim Sanskrit text.

Comparative Classical View: Synthesizes divergent opinions across the Brihat-trayi (e.g., Charaka's plain cold milk vs. Sushruta's sweet-liquid diet in Month 1; Charaka's ghee vs. Sushruta's Gokshura Ghrita in Month 6).

Bilingual & Responsive: Optimized for desktop and mobile browsers, ensuring access for rural healthcare workers and students.

Ethical & Clinical Guardrails: Prominently alerts users that AyurMitra is an academic research and clinical decision-support tool, not an emergency obstetric replacement.

📁 Repository Structure
Plaintext
ayurmitra/
├── LICENSE
├── README.md
├── frontend/
│   └── index.html                # Responsive clinical UI (React + Tailwind CDN)
└── backend/
    ├── app.py                    # FastAPI server & AyurParam orchestration
    ├── ingest.py                 # Vector embedding & FAISS indexer
    ├── requirements.txt          # Python dependencies
    ├── ayurmitra_faiss.index     # Generated FAISS vector database
    ├── chunks_metadata.json      # Indexed metadata and classical passages
    └── data/
        └── masanumasika_paricharya.json  # Verified Ayurvedic maternal corpus
🚀 Quickstart & Local Setup
1. Prerequisites
Python 3.10+ (tested on Python 3.12)

Windows / macOS / Linux

2. Backend Setup
Bash
cd backend
pip install -r requirements.txt
pip install fastapi uvicorn

# Build FAISS vector database
python ingest.py

# Launch FastAPI local server
python -m uvicorn app:app --reload --port 8000
API Documentation will be available at http://127.0.0.1:8000/docs.

3. Frontend Setup
Open frontend/index.html directly in any modern web browser, or serve it using any static server:

Bash
cd frontend
python -m http.server 3000
Navigate to http://127.0.0.1:3000.

⚖️ Ethical & Medical Disclaimer
AyurMitra is designed purely for academic, scholarly, and decision-support purposes. It is not intended for patient self-diagnosis or management of acute obstetric complications. Modern obstetric emergency evaluation should always take precedence in critical clinical situations.