"""Resume <-> Job-Description matching engine.

Combines several signals, most of them computed by from-scratch code in core/:
  * semantic similarity (pretrained embedding model) — one signal,
  * **from-scratch TF-IDF cosine** between resume and JD (core/tfidf.py),
  * explicit skill overlap (word-boundary matching),
  * **from-scratch skill co-occurrence graph** suggesting transferable skills
    to close the gap (core/skillgraph.py).

The blend can optionally be replaced by a **from-scratch logistic-regression**
scorer (core/logreg.py) trained on labeled pairs — see train_scorer().
CPU-only, no API key required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from core.tfidf import TfidfVectorizer
from core.skillgraph import SkillGraph

# A pragmatic, extensible skill vocabulary. Add to this freely.
SKILLS: Dict[str, List[str]] = {
    "python": ["python"],
    "javascript": ["javascript", "js", "node", "nodejs"],
    "typescript": ["typescript", "ts"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "sql": ["sql", "postgres", "postgresql", "mysql"],
    "react": ["react", "reactjs"],
    "aws": ["aws", "amazon web services"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn"],
    "deep learning": ["deep learning", "pytorch", "tensorflow", "neural network"],
    "nlp": ["nlp", "natural language processing"],
    "data analysis": ["data analysis", "pandas", "numpy", "data analytics"],
    "git": ["git", "github", "gitlab"],
    "rest api": ["rest", "rest api", "restful", "api"],
    "communication": ["communication", "stakeholder", "presentation"],
    "leadership": ["leadership", "led a team", "managed", "mentored"],
    "agile": ["agile", "scrum", "kanban"],
    "cloud": ["cloud", "gcp", "azure"],
}

# A tiny built-in co-occurrence corpus so the skill graph has something to
# reason over out of the box (real deployments would build this from postings).
_COOCCUR = [
    {"python", "machine learning", "deep learning", "nlp", "data analysis"},
    {"python", "docker", "kubernetes", "aws", "cloud"},
    {"javascript", "typescript", "react", "rest api"},
    {"python", "sql", "data analysis"},
    {"aws", "docker", "kubernetes", "git", "agile"},
    {"machine learning", "deep learning", "python", "git"},
    {"java", "sql", "rest api", "git"},
    {"communication", "leadership", "agile"},
]


@dataclass
class MatchResult:
    score: float                      # 0-100 overall match
    matched: List[str]                # skills present in BOTH
    missing: List[str]                # skills in the JD but NOT the resume
    extra: List[str]                  # skills in the resume not asked for
    verdict: str
    suggestions: List[str] = field(default_factory=list)  # transferable skills
    signals: Dict[str, float] = field(default_factory=dict)  # the component scores


def _find_skills(text: str) -> Set[str]:
    low = text.lower()
    found = set()
    for canonical, aliases in SKILLS.items():
        for a in aliases:
            # word-boundary match so "java" doesn't match "javascript"
            if re.search(r"(?<![a-z0-9+])" + re.escape(a) + r"(?![a-z0-9+])", low):
                found.add(canonical)
                break
    return found


class Matcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        # build the from-scratch skill graph once
        self.skill_graph = SkillGraph().build(_COOCCUR)

    def match(self, resume: str, jd: str) -> MatchResult:
        if not resume.strip() or not jd.strip():
            return MatchResult(0.0, [], [], [], "Paste both a resume and a job description.")

        # --- signal 1: semantic similarity (pretrained embeddings) ---
        embs = self.model.encode(
            [resume, jd], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        semantic_pct = max(0.0, float(embs[0] @ embs[1])) * 100

        # --- signal 2: from-scratch TF-IDF cosine ---
        vec = TfidfVectorizer(ngram=2)
        tfidf = vec.fit_transform([resume, jd])
        tfidf_pct = max(0.0, float(tfidf[0] @ tfidf[1])) * 100

        # --- signal 3: explicit skill overlap ---
        r_skills = _find_skills(resume)
        j_skills = _find_skills(jd)
        matched = sorted(r_skills & j_skills)
        missing = sorted(j_skills - r_skills)
        extra = sorted(r_skills - j_skills)
        coverage = (len(matched) / len(j_skills) * 100) if j_skills else semantic_pct

        # --- blend the three signals ---
        score = round(0.45 * semantic_pct + 0.20 * tfidf_pct + 0.35 * coverage, 1)

        # --- skill-graph suggestions: transferable skills toward the gaps ---
        suggestions = [s for s, _ in self.skill_graph.suggest(r_skills, top=5) if s in missing]
        if not suggestions:  # fall back to general adjacency
            suggestions = [s for s, _ in self.skill_graph.suggest(r_skills, top=3)]

        if score >= 75:
            verdict = "Strong match — you're well aligned with this role."
        elif score >= 55:
            verdict = "Decent match — address the gaps below to strengthen it."
        else:
            verdict = "Weak match — significant gaps relative to the role."

        return MatchResult(
            score, matched, missing, extra, verdict,
            suggestions=suggestions,
            signals={"semantic": round(semantic_pct, 1), "tfidf": round(tfidf_pct, 1), "skill_coverage": round(coverage, 1)},
        )
