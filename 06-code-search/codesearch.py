"""Semantic code search.

Index a folder of source files at the *function/class* level, then search by
natural language ("where do we validate the auth token?") and get the most
relevant code blocks back — ranked by meaning, not keywords.

Function extraction is language-aware for Python (via the `ast` module) and
falls back to a brace/indentation-based heuristic for other languages. Embedding
is done with a small Sentence-Transformers model, so it runs on CPU.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

CODE_EXTS = {".py", ".js", ".ts", ".java", ".go", ".rb", ".rs", ".c", ".cpp", ".cs"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}


@dataclass
class CodeBlock:
    path: str
    name: str
    start_line: int
    code: str


def _extract_python(path: str, source: str) -> List[CodeBlock]:
    blocks: List[CodeBlock] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return blocks
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            code = "\n".join(lines[start:end])
            blocks.append(CodeBlock(path, node.name, node.lineno, code[:2000]))
    return blocks


def _extract_generic(path: str, source: str) -> List[CodeBlock]:
    """Heuristic: treat top-level brace/`function`/`def`-ish blocks as units.

    Good enough to make non-Python files searchable; not a real parser.
    """
    blocks: List[CodeBlock] = []
    lines = source.splitlines()
    keywords = ("function ", "def ", "class ", "func ", "public ", "private ", "fn ")
    current: List[str] = []
    start_line = 1
    name = "block"
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if any(stripped.startswith(k) for k in keywords):
            if current:
                blocks.append(CodeBlock(path, name, start_line, "\n".join(current)[:2000]))
            current = [line]
            start_line = i
            name = stripped[:60]
        else:
            current.append(line)
    if current:
        blocks.append(CodeBlock(path, name, start_line, "\n".join(current)[:2000]))
    return blocks


def extract_blocks(path: str) -> List[CodeBlock]:
    try:
        source = open(path, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    if path.endswith(".py"):
        blocks = _extract_python(path, source)
        if blocks:
            return blocks
    return _extract_generic(path, source)


class CodeIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.blocks: List[CodeBlock] = []
        self._emb: Optional[np.ndarray] = None

    def index_folder(self, folder: str) -> int:
        self.blocks = []
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if os.path.splitext(fn)[1] in CODE_EXTS:
                    self.blocks.extend(extract_blocks(os.path.join(dirpath, fn)))
        if not self.blocks:
            self._emb = None
            return 0
        # Embed a "name + code" string so both signal contribute.
        texts = [f"{b.name}\n{b.code}" for b in self.blocks]
        self._emb = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return len(self.blocks)

    def search(self, query: str, k: int = 5) -> List[CodeBlock]:
        if self._emb is None or not self.blocks:
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = self._emb @ q[0]
        top = np.argsort(-scores)[:k]
        return [self.blocks[i] for i in top]
