"""FastAPI service exposing the PII vault.

WHY this module exists
----------------------
The core package is a library; this is the network boundary that turns it into
a service. It provides four endpoints:

  * ``GET  /health``      -- liveness probe + vault stats.
  * ``POST /redact``      -- detect PII in text, apply a redaction policy,
                             return redacted text + detections + issued tokens.
  * ``POST /detokenize``  -- reverse TOKENIZE surrogates back to originals via
                             the vault, gated by an API key (auth concept).
  * ``POST /risk-score``  -- run k-anonymity (+ optional l-diversity) over a
                             table of quasi-identifiers.

Design choices:
  * Pydantic models validate every request and shape every response, so the
    JSON contract is explicit and self-documenting in OpenAPI.
  * A single process-wide :class:`Vault` holds the reversible mappings; its
    key is derived from the ``VAULT_PASSPHRASE`` env var (a real deployment
    would inject this via a secrets manager). The detokenize endpoint requires
    a matching ``X-API-Key`` header -- a deliberately simple stand-in for real
    auth to demonstrate that detokenization is privileged.
  * Errors return structured JSON with appropriate HTTP status codes.

SECURITY CAVEAT: see the core.tokenize module docstring. The vault stores
plaintext originals in memory; do not run this as-is on real PII without an
encrypted, access-controlled backing store.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from core.detect import Detection, Detector, PIIType
from core.policy import Policy, PolicyEngine
from core.risk import k_anonymity
from core.tokenize import Vault, derive_key

# ---------------------------------------------------------------------------
# Process-wide singletons. The passphrase and API key come from the
# environment so secrets never live in code.
# ---------------------------------------------------------------------------
_PASSPHRASE = os.environ.get("VAULT_PASSPHRASE", "dev-insecure-passphrase")
_API_KEY = os.environ.get("VAULT_API_KEY", "dev-detokenize-key")

_VAULT = Vault(secret_key=derive_key(_PASSPHRASE))
_ENGINE = PolicyEngine(vault=_VAULT)
_DETECTOR = Detector()

app = FastAPI(
    title="privacy-pii-vault",
    version="1.0.0",
    description=(
        "PII detection, reversible format-preserving tokenization, redaction "
        "policies, and k-anonymity risk scoring -- implemented from scratch."
    ),
)


# ---------------------------------------------------------------------------
# Pydantic request/response models.
# ---------------------------------------------------------------------------
class DetectionModel(BaseModel):
    """Serialized form of a core :class:`Detection`."""

    type: str = Field(..., description="PII category, e.g. EMAIL")
    start: int = Field(..., ge=0, description="start char offset (inclusive)")
    end: int = Field(..., ge=0, description="end char offset (exclusive)")
    value: str = Field(..., description="the matched substring")
    confidence: float = Field(..., ge=0.0, le=1.0)

    @classmethod
    def from_core(cls, d: Detection) -> "DetectionModel":
        return cls(
            type=d.type.value,
            start=d.start,
            end=d.end,
            value=d.value,
            confidence=round(d.confidence, 4),
        )


class IssuedTokenModel(BaseModel):
    """A reversible token returned to the caller for later detokenization."""

    type: str
    start: int
    end: int
    token: str


class RedactRequest(BaseModel):
    """Body for ``POST /redact``."""

    text: str = Field(..., min_length=1, description="raw text to scan")
    default_action: str = Field(
        "REDACT",
        description="action for types without an explicit rule "
        "(REDACT|MASK|TOKENIZE|HASH|ALLOW)",
    )
    policy: Dict[str, str] = Field(
        default_factory=dict,
        description="per-type action overrides, e.g. {'EMAIL': 'TOKENIZE'}",
    )
    mask_keep: int = Field(4, ge=0, le=8, description="chars MASK leaves visible")
    min_confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="drop detections below this score"
    )


class RedactResponse(BaseModel):
    """Response for ``POST /redact``."""

    redacted_text: str
    detections: List[DetectionModel]
    issued_tokens: List[IssuedTokenModel]
    num_detections: int


class DetokenizeItem(BaseModel):
    """One token to reverse."""

    type: str = Field(..., description="PII type the token was issued under")
    token: str = Field(..., description="the surrogate to reverse")


class DetokenizeRequest(BaseModel):
    """Body for ``POST /detokenize``."""

    tokens: List[DetokenizeItem] = Field(..., min_length=1)


class DetokenizeResultItem(BaseModel):
    """Reverse-lookup outcome for a single token."""

    type: str
    token: str
    original: Optional[str]
    found: bool


class DetokenizeResponse(BaseModel):
    """Response for ``POST /detokenize``."""

    results: List[DetokenizeResultItem]


class RiskRequest(BaseModel):
    """Body for ``POST /risk-score``."""

    rows: List[Dict[str, object]] = Field(..., min_length=1)
    quasi_identifiers: List[str] = Field(..., min_length=1)
    threshold: int = Field(2, ge=1)
    sensitive_attrs: List[str] = Field(default_factory=list)


class RiskResponse(BaseModel):
    """Response for ``POST /risk-score``."""

    k: int
    threshold: int
    is_k_anonymous: bool
    num_classes: int
    flagged_rows: List[int]
    class_sizes: Dict[str, int]
    l_diversity: Dict[str, int]
    smallest_classes: List[List[object]]


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str
    version: str
    vault_entries: int


# ---------------------------------------------------------------------------
# Endpoints.
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe and basic vault statistics."""
    return HealthResponse(
        status="ok", version=app.version, vault_entries=_VAULT.size()
    )


@app.post("/redact", response_model=RedactResponse)
def redact(req: RedactRequest) -> RedactResponse:
    """Detect PII and apply the requested redaction policy.

    Returns the transformed text, every detection with its span and
    confidence, and any reversible tokens issued (so the caller can later
    detokenize them).
    """
    try:
        policy = Policy.from_mapping(
            req.policy, default_action=req.default_action, mask_keep=req.mask_keep
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid policy: {exc}",
        ) from exc

    detector = Detector(min_confidence=req.min_confidence)
    detections = detector.detect(req.text)
    result = _ENGINE.apply(req.text, detections, policy)

    return RedactResponse(
        redacted_text=result.redacted_text,
        detections=[DetectionModel.from_core(d) for d in result.detections],
        issued_tokens=[
            IssuedTokenModel(
                type=t.type.value,
                start=t.original_span[0],
                end=t.original_span[1],
                token=t.token,
            )
            for t in result.issued_tokens
        ],
        num_detections=len(result.detections),
    )


@app.post("/detokenize", response_model=DetokenizeResponse)
def detokenize(
    req: DetokenizeRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> DetokenizeResponse:
    """Reverse TOKENIZE surrogates to their originals (privileged).

    Requires a valid ``X-API-Key`` header -- detokenization is the sensitive
    operation that re-exposes raw PII, so it is gated behind auth.
    """
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key",
        )

    results: List[DetokenizeResultItem] = []
    for item in req.tokens:
        try:
            pii_type = PIIType(item.type.upper())
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown PII type: {item.type}",
            ) from exc
        original = _VAULT.detokenize(pii_type, item.token)
        results.append(
            DetokenizeResultItem(
                type=item.type.upper(),
                token=item.token,
                original=original,
                found=original is not None,
            )
        )
    return DetokenizeResponse(results=results)


@app.post("/risk-score", response_model=RiskResponse)
def risk_score(req: RiskRequest) -> RiskResponse:
    """Compute k-anonymity (and optional l-diversity) over a table."""
    try:
        report = k_anonymity(
            rows=req.rows,
            quasi_identifiers=req.quasi_identifiers,
            threshold=req.threshold,
            sensitive_attrs=req.sensitive_attrs or None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return RiskResponse(
        k=report.k,
        threshold=report.threshold,
        is_k_anonymous=report.is_k_anonymous,
        num_classes=report.num_classes,
        flagged_rows=report.flagged_rows,
        class_sizes=report.class_sizes,
        l_diversity=report.l_diversity,
        smallest_classes=[[label, size] for label, size in report.smallest_classes],
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
