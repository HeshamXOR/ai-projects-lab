"""FastAPI service for document intelligence.

Exposes the from-scratch extraction pipeline over HTTP:

* ``POST /extract`` -- accepts ``{"text": ..., "locale_hint": ...}`` and returns
  the structured invoice JSON with per-field confidence.
* ``GET  /health`` -- liveness probe.
* ``GET  /``        -- a tiny zero-dependency HTML demo (paste text, see JSON).

Run with::

    uvicorn app:app --reload

The pipeline itself (``core/``) is pure standard library; FastAPI / Pydantic
are only the transport layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from core import __version__
from core.pipeline import extract_document

app = FastAPI(
    title="Doc Intelligence",
    version=__version__,
    description="Convert invoice / receipt text into structured JSON, "
    "from-scratch extraction core.",
)


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #


class ExtractRequest(BaseModel):
    """Request body for ``POST /extract``."""

    text: str = Field(
        ...,
        description="Raw or OCR'd invoice / receipt text. Line breaks and "
        "column spacing are used as layout hints.",
        min_length=1,
    )
    locale_hint: Optional[str] = Field(
        default=None,
        description="Optional locale, e.g. 'en_US' or 'de_DE'. Disambiguates "
        "numeric dates and seeds a default currency.",
    )


class LineItemModel(BaseModel):
    description: str
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    line_total: Optional[str] = None
    currency: Optional[str] = None
    arithmetic_ok: bool = False
    confidence: float = 0.0


class ExtractResponse(BaseModel):
    """Structured extraction result."""

    vendor: Optional[str] = None
    invoice_number: Optional[str] = None
    date: Optional[str] = Field(default=None, description="ISO-8601 date")
    currency: Optional[str] = None
    subtotal: Optional[str] = None
    tax: Optional[str] = None
    total: Optional[str] = None
    line_items: List[LineItemModel] = Field(default_factory=list)
    confidence: Dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", version=__version__)


@app.post("/extract", response_model=ExtractResponse, tags=["extract"])
def extract(req: ExtractRequest) -> ExtractResponse:
    """Extract structured fields from invoice / receipt *text*."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="`text` must not be empty.")
    try:
        result = extract_document(text, locale_hint=req.locale_hint)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")
    return ExtractResponse(**result.to_dict())


_DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Doc Intelligence demo</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem;
         max-width: 1100px; margin-inline: auto; }
  h1 { font-size: 1.4rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  textarea { width: 100%; height: 420px; font-family: ui-monospace, monospace;
             font-size: 0.85rem; padding: 0.6rem; box-sizing: border-box; }
  pre { background: #1115; padding: 0.8rem; border-radius: 6px; overflow: auto;
        height: 420px; font-size: 0.8rem; }
  button { padding: 0.6rem 1.2rem; font-size: 1rem; cursor: pointer;
           margin-top: 0.6rem; }
  label { font-weight: 600; display:block; margin-bottom: 0.3rem; }
  .row { margin-bottom: 0.5rem; }
  input[type=text]{ padding: 0.4rem; width: 200px; }
  @media (max-width: 800px){ .grid { grid-template-columns: 1fr; }
    textarea, pre { height: 260px; } }
</style>
</head>
<body>
  <h1>Doc Intelligence &mdash; invoice &rarr; structured JSON</h1>
  <p>Paste invoice or receipt text and extract structured fields with
     per-field confidence. The extraction core is implemented from scratch.</p>
  <div class="row">
    <label for="locale">Locale hint (optional)</label>
    <input id="locale" type="text" placeholder="en_US" />
  </div>
  <div class="grid">
    <div>
      <label for="in">Invoice text</label>
      <textarea id="in" placeholder="Paste invoice text here..."></textarea>
    </div>
    <div>
      <label for="out">Result</label>
      <pre id="out">{ }</pre>
    </div>
  </div>
  <button id="go">Extract</button>
<script>
const sample = `ACME Web Solutions LLC
123 Market Street, Suite 400
San Francisco, CA 94103

INVOICE

Invoice #: INV-2024-0099
Invoice Date: 03/15/2024

Description                Qty    Unit Price    Amount
Web design services        10     150.00        1500.00
Annual hosting             1      240.00        240.00
Consulting                 5      $200.00       $1,000.00

Subtotal                                        $2,740.00
Tax (8.5%)                                       $232.90
Amount Due                                       $2,972.90`;
document.getElementById('in').value = sample;
async function run(){
  const text = document.getElementById('in').value;
  const locale = document.getElementById('locale').value || null;
  const out = document.getElementById('out');
  out.textContent = 'extracting...';
  try {
    const r = await fetch('/extract', {
      method:'POST', headers:{'content-type':'application/json'},
      body: JSON.stringify({text, locale_hint: locale})
    });
    const j = await r.json();
    out.textContent = JSON.stringify(j, null, 2);
  } catch(e){ out.textContent = 'Error: ' + e; }
}
document.getElementById('go').addEventListener('click', run);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["demo"])
def demo() -> str:
    """Serve the interactive HTML demo."""
    return _DEMO_HTML
