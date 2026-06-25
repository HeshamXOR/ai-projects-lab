"""FastAPI app exposing the from-scratch moderation pipeline.

Endpoints:
    GET  /health           -- liveness + model metadata.
    POST /moderate         -- moderate a single text.
    POST /moderate/batch   -- moderate a list of texts.
    POST /moderate/stream  -- stream per-chunk verdicts as NDJSON.
    GET  /                 -- minimal HTML demo page.

The moderation core is entirely standard-library; only this API layer needs
FastAPI / uvicorn / pydantic (see requirements.txt).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.pipeline import ModerationPipeline, ModerationResult

# Build the pipeline once at startup (trains the bundled Naive Bayes model).
pipeline = ModerationPipeline()

app = FastAPI(
    title="Real-time Text Moderation API",
    version="1.0.0",
    description=(
        "Streaming text moderation with a from-scratch policy engine and "
        "multinomial Naive Bayes classifier."
    ),
)


# --------------------------------------------------------------------------- #
# Pydantic request / response models.
# --------------------------------------------------------------------------- #
class ModerateRequest(BaseModel):
    """Request body for single-text moderation."""

    text: str = Field(..., min_length=1, max_length=20_000,
                      description="The text to moderate.")


class BatchRequest(BaseModel):
    """Request body for batch moderation."""

    texts: List[str] = Field(..., min_length=1, max_length=1_000,
                             description="A list of texts to moderate.")


class StreamRequest(BaseModel):
    """Request body for streaming moderation."""

    chunks: List[str] = Field(..., min_length=1, max_length=10_000,
                              description="Ordered text chunks to moderate.")
    cumulative: bool = Field(
        default=False,
        description="If true, each verdict covers all text seen so far.",
    )


def _result_payload(result: ModerationResult) -> Dict[str, object]:
    """Convert a :class:`ModerationResult` into a JSON-serializable dict."""
    return result.to_dict()


# --------------------------------------------------------------------------- #
# Endpoints.
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, object]:
    """Liveness probe with model metadata."""
    return {
        "status": "ok",
        "classes": pipeline.classifier.classes_,
        "vocabulary_size": len(pipeline.classifier.vocabulary_),
        "rules": len(pipeline.policy.rules),
        "thresholds": {
            "flag": pipeline.flag_threshold,
            "block": pipeline.block_threshold,
        },
    }


@app.post("/moderate")
def moderate(req: ModerateRequest) -> Dict[str, object]:
    """Moderate a single text and return decision + explanation."""
    try:
        result = pipeline.moderate(req.text)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _result_payload(result)


@app.post("/moderate/batch")
def moderate_batch(req: BatchRequest) -> Dict[str, object]:
    """Moderate a list of texts, preserving order."""
    try:
        results = pipeline.moderate_batch(req.texts)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"results": [_result_payload(r) for r in results]}


@app.post("/moderate/stream")
def moderate_stream(req: StreamRequest) -> StreamingResponse:
    """Stream one NDJSON verdict line per chunk via ``StreamingResponse``.

    Each line is a complete JSON object, so clients can parse incrementally.
    """

    def generate():
        for idx, result in enumerate(
            pipeline.moderate_stream(req.chunks, cumulative=req.cumulative)
        ):
            payload = _result_payload(result)
            payload["chunk_index"] = idx
            yield json.dumps(payload) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


_DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Real-time Moderation Demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  textarea { width: 100%; height: 120px; font-size: 1rem; padding: .5rem; }
  button { font-size: 1rem; padding: .5rem 1rem; margin-top: .5rem; cursor: pointer; }
  pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; border-radius: 6px; }
  .allow { color: #137333; } .flag { color: #b06000; } .block { color: #c5221f; }
</style>
</head>
<body>
  <h1>Real-time Text Moderation</h1>
  <p>From-scratch policy engine + multinomial Naive Bayes. Type text and moderate.</p>
  <textarea id="t" placeholder="Enter text to moderate..."></textarea>
  <br/>
  <button onclick="run()">Moderate</button>
  <h3 id="verdict"></h3>
  <pre id="out"></pre>
<script>
async function run() {
  const text = document.getElementById('t').value;
  const res = await fetch('/moderate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  const data = await res.json();
  const d = data.decision;
  const v = document.getElementById('verdict');
  v.textContent = 'Decision: ' + d.toUpperCase() + ' (overall ' + data.score.overall + ')';
  v.className = d;
  document.getElementById('out').textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def demo() -> str:
    """Serve a minimal interactive demo page."""
    return _DEMO_HTML
