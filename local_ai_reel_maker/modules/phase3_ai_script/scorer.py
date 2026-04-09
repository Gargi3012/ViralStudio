from __future__ import annotations
import json
import logging
import argparse
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

# Set HF_HOME environment variable to cache models locally
import os
os.environ["HF_HOME"] = str(Path("models/huggingface").resolve())
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def setup_logger(name: str, log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────
#  Viral Keyword Lexicon
# ─────────────────────────────────────────────

VIRAL_KEYWORDS: dict[str, float] = {
    "secret": 1.0, "truth": 1.0, "revealed": 1.0, "shocking": 1.0,
    "nobody": 1.0, "everybody": 1.0, "changed": 1.0, "transform": 1.0,
    "breakthrough": 1.0, "incredible": 1.0, "unbelievable": 1.0, "mindset": 1.0,
    "success": 0.8, "failure": 0.8, "millionaire": 0.8, "freedom": 0.8,
    "fear": 0.8, "pain": 0.8, "discipline": 0.8, "sacrifice": 0.8,
    "winner": 0.8, "loser": 0.8, "rich": 0.8, "poor": 0.8, "hustle": 0.8,
    "grind": 0.8, "passion": 0.8, "money": 0.5, "dream": 0.5, "goal": 0.5,
    "focus": 0.5, "growth": 0.5, "hard": 0.5, "easy": 0.5, "life": 0.5,
    "power": 0.5, "energy": 0.5, "motivation": 0.5, "inspire": 0.5,
    "action": 0.5, "believe": 0.5, "achieve": 0.5, "courage": 0.5,
    "confidence": 0.5, "habit": 0.5,
}


# ─────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────

@dataclass
class SemanticChunk:
    chunk_id:          int
    segment_ids:       list[int]
    text:              str
    start:             float
    end:               float
    duration:          float
    avg_db:            float = 0.0
    max_db:            float = 0.0
    sentiment_score:   float = 0.0
    audio_intensity:   float = 0.0
    keyword_density:   float = 0.0
    viral_score:       float = 0.0
    sentiment_label:   str   = ""
    top_keywords:      list[str] = field(default_factory=list)
    word_count:        int = 0


@dataclass
class ViralClip:
    rank:              int
    chunk_id:          int
    start:             float
    end:               float
    duration:          float
    text:              str
    viral_score:       float
    sentiment_score:   float
    audio_intensity:   float
    keyword_density:   float
    sentiment_label:   str
    top_keywords:      list[str]
    avg_db:            float
    max_db:            float
    word_count:        int


@dataclass
class ScoringResult:
    session_id:        str
    transcript_path:   str
    total_chunks:      int
    top_clips:         list[dict]
    model_sentence:    str
    model_sentiment:   str
    weights:           dict
    processing_notes:  list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
#  1. Semantic Chunker
# ─────────────────────────────────────────────

class SemanticChunker:
    SENTENCE_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        min_duration: float = 25.0,
        max_duration: float = 65.0,
        similarity_threshold: float = 0.45,
        logger: Optional[logging.Logger] = None,
    ):
        self.min_duration         = min_duration
        self.max_duration         = max_duration
        self.similarity_threshold = similarity_threshold
        self.logger               = logger or logging.getLogger(__name__)
        self._model               = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.SENTENCE_MODEL)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def chunk(self, segments: list[dict]) -> list[list[dict]]:
        if not segments: return []
        if len(segments) == 1: return [segments]

        texts = [s.get("text", "").strip() for s in segments]
        embeddings = self._embed(texts)
        
        sims = []
        for i in range(len(embeddings) - 1):
            sim = 1.0 - float(cosine_distance(embeddings[i], embeddings[i+1]))
            sims.append(sim)
        
        boundaries = [0]
        for i, sim in enumerate(sims):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(segments))

        raw_chunks = []
        for i in range(len(boundaries) - 1):
            raw_chunks.append(segments[boundaries[i]: boundaries[i+1]])

        return self._enforce_duration(raw_chunks, np.array(sims))

    def _duration(self, segs: list[dict]) -> float:
        if not segs: return 0.0
        return float(segs[-1]["end"]) - float(segs[0]["start"])

    def _enforce_duration(self, chunks: list[list[dict]], sims: np.ndarray) -> list[list[dict]]:
        res = [c[:] for c in chunks]
        for _ in range(5):
            changed = False
            # Merge
            merged = []
            i = 0
            while i < len(res):
                c = res[i]
                if self._duration(c) < self.min_duration and i + 1 < len(res):
                    if self._duration(c + res[i+1]) <= self.max_duration * 1.2:
                        merged.append(c + res[i+1])
                        i += 2
                        changed = True
                        continue
                merged.append(c)
                i += 1
            res = merged
            # Split
            split = []
            for c in res:
                if self._duration(c) > self.max_duration and len(c) > 1:
                    best_s, best_sim = len(c)//2, 2.0
                    for j in range(1, len(c)):
                        idx = min(c[j].get("segment_id", j), len(sims)-1)
                        if sims[idx] < best_sim:
                            best_sim, best_s = sims[idx], j
                    split.append(c[:best_s])
                    split.append(c[best_s:])
                    changed = True
                else:
                    split.append(c)
            res = split
            if not changed: break
        return res


# ─────────────────────────────────────────────
#  2. Sentiment Scorer
# ─────────────────────────────────────────────

class SentimentScorer:
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("sentiment-analysis", model=self.SENTIMENT_MODEL, device=-1)
        return self._pipe

    def score_text(self, text: str) -> dict:
        if not text.strip(): return {"score": 0.5, "label": "NEUTRAL"}
        try:
            res = self._load()(text[:1500])[0]
            score = float(res["score"])
            if res["label"] == "NEGATIVE": score = 1.0 - score
            return {"score": score, "label": res["label"]}
        except:
            return {"score": 0.5, "label": "UNKNOWN"}


# ─────────────────────────────────────────────
#  3. Main Engine
# ─────────────────────────────────────────────

class ScoringEngine:
    WEIGHTS = {"sentiment": 0.4, "audio": 0.4, "keyword": 0.2}

    def __init__(self, min_duration=25.0, max_duration=65.0, similarity_threshold=0.45, top_n=5):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.similarity_threshold = similarity_threshold
        self.top_n = top_n
        self.logger = logging.getLogger(__name__)

    def _combine_scores(self, sent, audio, keyword, w_sent, w_audio, w_keyword):
        return float(np.clip(sent*w_sent + audio*w_audio + keyword*w_keyword, 0, 1))

    def _save_result(self, result: ScoringResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "viral_clips.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        return path

    def run(self, transcript_path: Path, output_dir: Path, audio_path: Optional[Path] = None) -> ScoringResult:
        session_id = transcript_path.stem.split("_transcript")[0]
        self.logger = setup_logger(f"scorer.{session_id}")
        
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", [])

        chunker = SemanticChunker(self.min_duration, self.max_duration, self.similarity_threshold, self.logger)
        raw_groups = chunker.chunk(segments)
        
        chunks = []
        for i, group in enumerate(raw_groups):
            text = " ".join(s["text"] for s in group).strip()
            chunks.append(SemanticChunk(
                chunk_id=i, segment_ids=[s.get("id",0) for s in group],
                text=text, start=group[0]["start"], end=group[-1]["end"],
                duration=group[-1]["end"]-group[0]["start"], word_count=len(text.split())
            ))

        sent_scorer = SentimentScorer(self.logger)
        for c in chunks:
            res = sent_scorer.score_text(c.text)
            c.sentiment_score, c.sentiment_label = res["score"], res["label"]
            
            # Keyword score
            words = c.text.lower().split()
            kw_hits = sum(VIRAL_KEYWORDS.get(w.strip(".,!"), 0) for w in words)
            c.keyword_density = min(kw_hits / (len(words)+1) / 0.1, 1.0)
            c.viral_score = self._combine_scores(c.sentiment_score, 0.5, c.keyword_density, 0.4, 0.4, 0.2)

        # Overlap Prevention
        chunks.sort(key=lambda x: x.viral_score, reverse=True)
        top_clips_obj: list[ViralClip] = []
        for cand in chunks:
            if len(top_clips_obj) >= self.top_n: break
            if any(abs(cand.start - s.start) < 45 for s in top_clips_obj): continue
            top_clips_obj.append(ViralClip(
                rank=len(top_clips_obj)+1, chunk_id=cand.chunk_id,
                start=cand.start, end=cand.end, duration=cand.duration, text=cand.text,
                viral_score=cand.viral_score, sentiment_score=cand.sentiment_score,
                audio_intensity=0.5, keyword_density=cand.keyword_density,
                sentiment_label=cand.sentiment_label, top_keywords=[],
                avg_db=0, max_db=0, word_count=cand.word_count
            ))

        result = ScoringResult(
            session_id=session_id, transcript_path=str(transcript_path),
            total_chunks=len(chunks), top_clips=[asdict(c) for c in top_clips_obj],
            model_sentence=SemanticChunker.SENTENCE_MODEL,
            model_sentiment=SentimentScorer.SENTIMENT_MODEL, weights=self.WEIGHTS
        )
        out_path = self._save_result(result, output_dir)
        self._print_summary(result, top_clips_obj, out_path)
        return result

    def _print_summary(self, result, clips, out_path):
        def p(s=""):
            try: print(s)
            except: print(s.encode('ascii','replace').decode('ascii'))
        p("\n" + "═"*60)
        p("  PHASE 3 — VIRAL SCORING (OVERLAP PROTECTED)")
        p("═"*60)
        for c in clips:
            p(f"  #{c.rank} | Score: {c.viral_score:.3f} | {c.start:.1f}s -> {c.end:.1f}s")
        p("═"*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()
    engine = ScoringEngine(top_n=args.top_n)
    engine.run(Path(args.transcript), Path(args.output_dir or "."))