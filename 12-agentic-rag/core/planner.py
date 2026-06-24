"""Query planner: decompose a complex question into sub-questions.

Multi-hop questions ("How did the company's revenue change after they acquired
X?") can't be answered by a single retrieval — they need to be broken into
steps. The planner splits a question into sub-questions that can each be
retrieved independently, then recombined.

This is a from-scratch, rule-based planner (conjunctions, comparatives, and
question words) with an optional LLM hook. Rule-based means it's deterministic
and testable; the LLM hook upgrades quality when a key is available.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional


def _rule_based_plan(question: str) -> List[str]:
    q = question.strip()
    parts: List[str] = []

    # split on explicit conjunctions that usually join separate asks
    for chunk in re.split(r"\b(?:and then|and also|and|then|;)\b", q, flags=re.I):
        c = chunk.strip(" ,.")
        if len(c.split()) >= 2:
            parts.append(c)

    # comparative questions need both sides retrieved
    m = re.search(r"(.+?)\b(?:compared to|versus|vs\.?|than)\b(.+)", q, flags=re.I)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        parts = [f"information about {a}", f"information about {b}"]

    # de-dup, preserve order; fall back to the whole question
    seen, plan = set(), []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            plan.append(p if p.endswith("?") else p + "?")
    return plan or [q]


def plan_query(question: str, llm: Optional[Callable[[str], str]] = None) -> List[str]:
    """Return a list of sub-questions. Uses the LLM if provided, else rules."""
    if llm is not None:
        prompt = (
            "Break this question into 1-4 standalone sub-questions, one per line, "
            f"no numbering:\n{question}"
        )
        try:
            text = llm(prompt)
            subs = [l.strip(" -•").strip() for l in text.splitlines() if l.strip()]
            subs = [s for s in subs if len(s.split()) >= 2]
            if subs:
                return subs[:4]
        except Exception:
            pass
    return _rule_based_plan(question)
