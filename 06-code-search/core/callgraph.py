"""Call graph from Python source via the AST — from scratch.

Parses Python and builds "who calls whom": for each function definition, which
other functions it invokes. Enables "find related functions" and "what calls
this?" — the structural backbone of code-navigation tools.

Uses the standard-library `ast` module (parsing) but the graph construction,
caller/callee resolution, and traversal are hand-written.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import Dict, List, Set


class CallGraph:
    def __init__(self):
        self.calls: Dict[str, Set[str]] = defaultdict(set)   # caller -> {callees}
        self.called_by: Dict[str, Set[str]] = defaultdict(set)
        self.defined: Set[str] = set()

    def add_source(self, source: str) -> None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = node.name
                self.defined.add(fn)
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        callee = self._callee_name(sub.func)
                        if callee:
                            self.calls[fn].add(callee)
                            self.called_by[callee].add(fn)

    @staticmethod
    def _callee_name(func_node) -> str:
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return ""

    def callees(self, fn: str) -> Set[str]:
        return {c for c in self.calls.get(fn, set()) if c in self.defined}

    def callers(self, fn: str) -> Set[str]:
        return self.called_by.get(fn, set())

    def related(self, fn: str) -> Set[str]:
        """Functions one hop away in either direction (callers ∪ callees)."""
        return self.callees(fn) | self.callers(fn)
