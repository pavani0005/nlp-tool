from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import re
import math
from collections import Counter

app = FastAPI(title="NLP Text Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    word_count: int
    sentence_count: int
    char_count: int
    char_no_spaces: int
    avg_word_length: float
    avg_sentence_length: float
    reading_time_seconds: float
    top_words: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
    readability: Dict[str, Any]
    pos_tags: Dict[str, int]
    unique_words: int
    lexical_diversity: float
    paragraphs: int
    longest_sentence: str
    shortest_sentence: str

# Simple sentiment word lists
POSITIVE_WORDS = set([
    "good","great","excellent","amazing","wonderful","fantastic","love","like","happy",
    "joy","beautiful","best","perfect","awesome","nice","brilliant","outstanding",
    "superb","positive","success","win","glad","pleased","delighted","excited",
    "enjoy","helpful","kind","generous","grateful","thankful","hope","bright","clear"
])
NEGATIVE_WORDS = set([
    "bad","terrible","awful","horrible","hate","dislike","sad","unhappy","poor",
    "worst","ugly","negative","fail","wrong","difficult","problem","issue","pain",
    "hurt","fear","angry","frustrated","disappointed","annoyed","bored","boring",
    "terrible","dreadful","nasty","harsh","cruel","suffer","loss","dark","doubt"
])

def tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def get_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def analyze_sentiment(words: List[str]) -> Dict[str, Any]:
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = len(words) or 1
    score = (pos - neg) / total
    if score > 0.02:
        label = "Positive"
        emoji = "😊"
        color = "#22c55e"
    elif score < -0.02:
        label = "Negative"
        emoji = "😞"
        color = "#ef4444"
    else:
        label = "Neutral"
        emoji = "😐"
        color = "#f59e0b"
    return {
        "label": label, "emoji": emoji, "color": color,
        "score": round(score, 4),
        "positive_count": pos, "negative_count": neg,
        "confidence": round(min(abs(score) * 20, 1.0), 2)
    }

def flesch_kincaid(text: str, words: List[str], sentences: List[str]) -> Dict[str, Any]:
    syllables = sum(count_syllables(w) for w in words)
    wc = len(words) or 1
    sc = len(sentences) or 1
    fre = 206.835 - 1.015*(wc/sc) - 84.6*(syllables/wc)
    fre = max(0, min(100, fre))
    if fre >= 90: level = "Very Easy"; grade = "5th grade"
    elif fre >= 80: level = "Easy"; grade = "6th grade"
    elif fre >= 70: level = "Fairly Easy"; grade = "7th grade"
    elif fre >= 60: level = "Standard"; grade = "8-9th grade"
    elif fre >= 50: level = "Fairly Difficult"; grade = "10-12th grade"
    elif fre >= 30: level = "Difficult"; grade = "College"
    else: level = "Very Difficult"; grade = "Professional"
    return {"score": round(fre, 1), "level": level, "grade": grade}

def count_syllables(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r'[aeiou]+', word))
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)

def simple_pos_tags(words: List[str]) -> Dict[str, int]:
    common_articles = {"a","an","the"}
    common_preps = {"in","on","at","to","for","of","with","by","from","about","as","into","through"}
    common_conj = {"and","or","but","so","yet","nor","for","although","because","since","while"}
    common_pronouns = {"i","me","my","we","our","you","your","he","she","it","they","them","his","her","its","their"}
    aux_verbs = {"is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","shall","should","may","might","must","can","could"}
    counts = {"Nouns": 0, "Verbs": 0, "Adjectives": 0, "Articles": 0, "Pronouns": 0, "Prepositions": 0, "Conjunctions": 0}
    adj_suffixes = ("ful","less","ous","ive","al","ic","able","ible","ent","ant")
    verb_suffixes = ("ing","ed","tion","ate","ize","ise","fy","en")
    noun_suffixes = ("tion","ness","ment","ity","er","or","ist","ism","ance","ence")
    for w in words:
        if w in common_articles: counts["Articles"] += 1
        elif w in common_preps: counts["Prepositions"] += 1
        elif w in common_conj: counts["Conjunctions"] += 1
        elif w in common_pronouns: counts["Pronouns"] += 1
        elif w in aux_verbs: counts["Verbs"] += 1
        elif any(w.endswith(s) for s in adj_suffixes): counts["Adjectives"] += 1
        elif any(w.endswith(s) for s in verb_suffixes): counts["Verbs"] += 1
        elif any(w.endswith(s) for s in noun_suffixes): counts["Nouns"] += 1
        else: counts["Nouns"] += 1
    return counts

@app.get("/")
def root():
    return {"status": "NLP Analysis API is running", "version": "1.0.0"}

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_text(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 50000:
        raise HTTPException(status_code=400, detail="Text too long (max 50,000 chars)")

    words = tokenize(text)
    sentences = get_sentences(text)
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])

    wc = len(words)
    sc = len(sentences)
    filtered = [w for w in words if len(w) > 2 and w not in {"the","and","for","are","but","not","you","all","can","her","was","one","our","out","had","his","has","him","how","did","its","let","may","nor","now","old","own","say","she","two","way","who"}]

    top_words = [{"word": w, "count": c, "percent": round(c/wc*100, 1)} for w, c in Counter(filtered).most_common(10)]
    avg_wl = round(sum(len(w) for w in words)/wc, 2) if wc else 0
    avg_sl = round(wc/sc, 1) if sc else 0
    reading_time = round(wc / 238 * 60, 1)  # ~238 WPM avg

    sorted_sentences = sorted(sentences, key=len)
    shortest = sorted_sentences[0] if sorted_sentences else ""
    longest = sorted_sentences[-1] if sorted_sentences else ""

    unique = len(set(words))
    diversity = round(unique/wc, 3) if wc else 0

    return AnalysisResponse(
        word_count=wc,
        sentence_count=sc,
        char_count=len(text),
        char_no_spaces=len(text.replace(" ", "")),
        avg_word_length=avg_wl,
        avg_sentence_length=avg_sl,
        reading_time_seconds=reading_time,
        top_words=top_words,
        sentiment=analyze_sentiment(words),
        readability=flesch_kincaid(text, words, sentences),
        pos_tags=simple_pos_tags(words),
        unique_words=unique,
        lexical_diversity=diversity,
        paragraphs=paragraphs,
        longest_sentence=longest[:200],
        shortest_sentence=shortest[:200]
    )

@app.post("/compare")
def compare_texts(texts: List[str]):
    if len(texts) < 2 or len(texts) > 4:
        raise HTTPException(status_code=400, detail="Provide 2-4 texts to compare")
    results = []
    for t in texts:
        words = tokenize(t)
        sentences = get_sentences(t)
        wc = len(words)
        sc = len(sentences) or 1
        results.append({
            "word_count": wc,
            "sentence_count": sc,
            "avg_sentence_length": round(wc/sc, 1),
            "sentiment": analyze_sentiment(words)["label"],
            "readability": flesch_kincaid(t, words, sentences)["level"],
            "lexical_diversity": round(len(set(words))/wc, 3) if wc else 0
        })
    return {"comparisons": results}
