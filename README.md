# 🔬 LexiLens — NLP Text Analysis Tool

A full-stack NLP text analysis application built with **FastAPI** (Python backend) and **Plain HTML/CSS/JS** (frontend).

---

## 📁 Project Structure

```
nlp-tool/
├── backend/
│   ├── main.py              # FastAPI backend (all NLP logic)
│   └── requirements.txt     # Python dependencies
└── frontend/
    └── index.html           # Single-file frontend (no build needed)
```

---

## 🚀 Quick Start

### 1. Set up the Backend

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```

The API will be available at: **http://localhost:8000**  
Interactive API docs at: **http://localhost:8000/docs**

### 2. Open the Frontend

Simply open the HTML file in your browser:

```bash
# Option A: direct open
open frontend/index.html

# Option B: serve locally with Python
cd frontend
python -m http.server 3000
# Then visit http://localhost:3000
```

---

## ✨ Features

### Text Metrics
- Word count, sentence count, character count
- Average word & sentence length
- Reading time estimation
- Paragraph count, unique word count

### Sentiment Analysis
- Positive / Neutral / Negative classification
- Sentiment score and confidence
- Positive vs. negative word counts

### Readability
- Flesch–Kincaid Reading Ease score (0–100)
- Reading level (Very Easy → Very Difficult)
- Grade level equivalent

### Linguistic Profile
- Top 10 keywords with frequency bars
- Part-of-speech breakdown (Nouns, Verbs, Adjectives, etc.)
- Lexical diversity score (unique vocab ratio)

### Notable Sentences
- Longest and shortest sentence extraction

---

## 🔌 API Endpoints

| Method | Endpoint   | Description                |
|--------|-----------|----------------------------|
| GET    | `/`        | Health check               |
| POST   | `/analyze` | Full NLP analysis          |
| POST   | `/compare` | Compare 2–4 texts          |

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text goes here. Paste anything you want to analyze."}'
```

### Example Response (abridged)

```json
{
  "word_count": 12,
  "sentence_count": 2,
  "char_count": 55,
  "sentiment": {
    "label": "Neutral",
    "score": 0.0,
    "emoji": "😐"
  },
  "readability": {
    "score": 72.4,
    "level": "Fairly Easy",
    "grade": "7th grade"
  },
  "top_words": [
    {"word": "paste", "count": 1, "percent": 8.3}
  ]
}
```

---

## 🛠 Tech Stack

| Layer    | Technology         |
|----------|--------------------|
| Backend  | Python + FastAPI   |
| NLP      | Pure Python (no heavy ML libs needed) |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Fonts    | Google Fonts (Syne, DM Mono, Lora) |

---

## 🔧 Extending the Project

### Add ML-powered sentiment (transformers)
```bash
pip install transformers torch
```
Then replace the `analyze_sentiment()` function in `main.py` with a HuggingFace pipeline:
```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
```

### Add spaCy for better NLP
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Add a database (SQLite/PostgreSQL)
Use `databases` + `SQLAlchemy` to store analysis history.

---

## 📄 License
MIT — free to use, extend, and deploy.
