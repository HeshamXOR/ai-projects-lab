"""Resume <-> Job-Description matching engine.

Computes a semantic match score and a skill-level gap analysis between a resume
and a job description. CPU-only, no API key. The skill list is a curated,
extensible set of common tech/business skills; matching is case-insensitive and
handles a few aliases (e.g. "js" -> "javascript").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

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


@dataclass
class MatchResult:
    score: float                      # 0-100 overall semantic match
    matched: List[str]                # skills present in BOTH
    missing: List[str]                # skills in the JD but NOT the resume
    extra: List[str]                  # skills in the resume not asked for
    verdict: str


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

    def match(self, resume: str, jd: str) -> MatchResult:
        if not resume.strip() or not jd.strip():
            return MatchResult(0.0, [], [], [], "Paste both a resume and a job description.")

        # Semantic similarity of the two documents as a whole.
        embs = self.model.encode(
            [resume, jd], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        semantic = float(embs[0] @ embs[1])  # cosine, in [-1, 1]
        semantic_pct = max(0.0, semantic) * 100

        # Skill overlap.
        r_skills = _find_skills(resume)
        j_skills = _find_skills(jd)
        matched = sorted(r_skills & j_skills)
        missing = sorted(j_skills - r_skills)
        extra = sorted(r_skills - j_skills)

        coverage = (len(matched) / len(j_skills) * 100) if j_skills else semantic_pct

        # Blend semantic similarity (60%) with explicit skill coverage (40%).
        score = round(0.6 * semantic_pct + 0.4 * coverage, 1)

        if score >= 75:
            verdict = "Strong match — you're well aligned with this role."
        elif score >= 55:
            verdict = "Decent match — address the gaps below to strengthen it."
        else:
            verdict = "Weak match — significant gaps relative to the role."

        return MatchResult(score, matched, missing, extra, verdict)
