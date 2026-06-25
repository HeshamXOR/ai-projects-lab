"""FastAPI service for the LoRA fine-tune studio.

Endpoints
---------
  GET  /health    -> liveness + which base model is loaded
  POST /finetune  -> kick a (tiny) LoRA training run on supplied records, return
                     param accounting + loss history + before/after perplexity
  POST /eval      -> perplexity + token accuracy on supplied records
  POST /generate  -> greedy/sampled generation from base (+ trained adapter)

Design notes
------------
* The base LLM is pluggable. We build a deterministic TinyCausalLM at startup so
  the service runs with zero downloads. `build_studio()` is the dependency-
  injection seam: swap in a real HF model (wrapped to the CausalLM contract) and
  every endpoint keeps working.
* Each /finetune call trains a *fresh* model+adapter so runs are independent and
  reproducible; the trained adapter is cached in-process under a returned
  `session_id` so a subsequent /generate can use it.
* All responses are structured JSON; all inputs are validated by Pydantic models;
  errors map to clean HTTP status codes via a handler.
"""

from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from core import (
    AlpacaFormatter,
    CharTokenizer,
    LoRATrainer,
    TinyCausalLM,
    TrainingConfig,
)
from core.data import build_training_batch

# ============================================================================
# Pydantic request/response schemas
# ============================================================================
class Record(BaseModel):
    """One instruction-tuning example."""

    instruction: str = Field(..., min_length=1, description="The task instruction.")
    input: str = Field("", description="Optional extra context for the task.")
    output: str = Field(..., min_length=1, description="The target response.")


class ConfigPatch(BaseModel):
    """Subset of TrainingConfig knobs a client may override."""

    lr: Optional[float] = None
    r: Optional[int] = None
    alpha: Optional[float] = None
    dropout: Optional[float] = None
    epochs: Optional[int] = Field(None, ge=1, le=50)
    batch_size: Optional[int] = Field(None, ge=1, le=128)
    max_seq_len: Optional[int] = Field(None, ge=1, le=2048)
    target_modules: Optional[List[str]] = None
    seed: Optional[int] = None

    def merged_config(self) -> TrainingConfig:
        data = {k: v for k, v in self.model_dump().items() if v is not None}
        return TrainingConfig.from_dict(data)


class FinetuneRequest(BaseModel):
    records: List[Record] = Field(..., min_length=1)
    config: ConfigPatch = Field(default_factory=ConfigPatch)
    eval_records: Optional[List[Record]] = None

    @field_validator("records")
    @classmethod
    def _non_empty(cls, v: List[Record]) -> List[Record]:
        if not v:
            raise ValueError("records must be non-empty")
        return v


class FinetuneResponse(BaseModel):
    session_id: str
    config: Dict[str, object]
    param_stats: Dict[str, int]
    steps: int
    initial_loss: Optional[float]
    final_loss: Optional[float]
    loss_improved: Optional[bool]
    perplexity_before: Optional[float]
    perplexity_after: Optional[float]


class EvalRequest(BaseModel):
    records: List[Record] = Field(..., min_length=1)
    session_id: Optional[str] = Field(
        None, description="Use the adapter from a prior /finetune; else base model."
    )


class EvalResponse(BaseModel):
    perplexity: float
    token_accuracy: float
    num_tokens: int
    mean_nll: float
    used_adapter: bool


class GenerateRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    input: str = ""
    max_new_tokens: int = Field(48, ge=1, le=512)
    greedy: bool = True
    temperature: float = Field(1.0, gt=0.0, le=5.0)
    session_id: Optional[str] = None


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    used_adapter: bool


# ============================================================================
# Studio: holds the base tokenizer/model factory + adapter session cache
# ============================================================================
class Studio:
    """Owns the pluggable base model factory and trained-adapter sessions."""

    def __init__(self) -> None:
        # A fixed corpus seeds a deterministic char vocab so /generate is stable
        # even before any fine-tuning. In production, replace with a real
        # tokenizer + model via the same two attributes.
        seed_corpus = (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n\n### Instruction:\n### Input:\n"
            "### Response:\nabcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ "
            "0123456789 .,!?;:'\"-()\n"
        )
        self.tokenizer = CharTokenizer.from_corpus([seed_corpus])
        self.formatter = AlpacaFormatter()
        self.model_name = "TinyCausalLM(d=64, layers=2)"
        # session_id -> trained model (with merged-or-live adapter)
        self._sessions: Dict[str, object] = {}

    def fresh_model(self, seed: int = 0) -> TinyCausalLM:
        """Construct a new base model. The DI seam for real HF models."""
        return TinyCausalLM(vocab_size=self.tokenizer.vocab_size, seed=seed)

    def store_session(self, model: object) -> str:
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = model
        return sid

    def get_session(self, sid: str):
        if sid not in self._sessions:
            raise KeyError(sid)
        return self._sessions[sid]

    def has_session(self, sid: Optional[str]) -> bool:
        return bool(sid) and sid in self._sessions


def build_studio() -> Studio:
    """Factory used at startup (and overridable in tests)."""
    return Studio()


# ============================================================================
# App wiring
# ============================================================================
app = FastAPI(
    title="finetune-studio",
    version="1.0.0",
    description="From-scratch LoRA fine-tuning, evaluation, and serving.",
)
STUDIO = build_studio()


def _clean_float(x: float) -> Optional[float]:
    """JSON can't hold NaN/inf; map them to None for clean responses."""
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


@app.exception_handler(ValueError)
async def _value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.get("/health")
def health() -> Dict[str, object]:
    """Liveness probe + which base model is loaded."""
    return {
        "status": "ok",
        "model": STUDIO.model_name,
        "vocab_size": STUDIO.tokenizer.vocab_size,
        "active_sessions": len(STUDIO._sessions),
    }


@app.post("/finetune", response_model=FinetuneResponse)
def finetune(req: FinetuneRequest) -> FinetuneResponse:
    """Run a small LoRA fine-tune on the provided records and report metrics."""
    cfg = req.config.merged_config()
    records = [r.model_dump() for r in req.records]

    model = STUDIO.fresh_model(seed=cfg.seed)
    trainer = LoRATrainer(model, cfg, STUDIO.tokenizer)

    eval_recs = (
        [r.model_dump() for r in req.eval_records] if req.eval_records else records
    )

    try:
        ppl_before = trainer.evaluate(eval_recs).perplexity
        summary = trainer.train(records)
        ppl_after = trainer.evaluate(eval_recs).perplexity
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    sid = STUDIO.store_session(model)
    return FinetuneResponse(
        session_id=sid,
        config=cfg.to_dict(),
        param_stats=trainer.param_stats(),
        steps=int(summary["steps"]),
        initial_loss=_clean_float(summary["initial_loss"]),
        final_loss=_clean_float(summary["final_loss"]),
        loss_improved=summary["loss_improved"],
        perplexity_before=_clean_float(ppl_before),
        perplexity_after=_clean_float(ppl_after),
    )


@app.post("/eval", response_model=EvalResponse)
def evaluate(req: EvalRequest) -> EvalResponse:
    """Perplexity + token accuracy on records, optionally using a trained adapter."""
    used_adapter = STUDIO.has_session(req.session_id)
    if req.session_id and not used_adapter:
        raise HTTPException(status_code=404, detail=f"unknown session {req.session_id}")

    model = STUDIO.get_session(req.session_id) if used_adapter else STUDIO.fresh_model()
    cfg = TrainingConfig()
    records = [r.model_dump() for r in req.records]

    from core.eval import Evaluator

    evaluator = Evaluator()
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        for r in records:
            batch = build_training_batch([r], STUDIO.tokenizer, cfg.max_seq_len, STUDIO.formatter)
            logits = model(batch["input_ids"])  # type: ignore[operator]
            evaluator.update(logits[:, :-1, :], batch["labels"][:, 1:])
    res = evaluator.result()

    return EvalResponse(
        perplexity=_clean_float(res.perplexity) or float("inf"),
        token_accuracy=_clean_float(res.token_accuracy) or 0.0,
        num_tokens=res.num_tokens,
        mean_nll=_clean_float(res.mean_nll) or 0.0,
        used_adapter=used_adapter,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate a completion, optionally from a fine-tuned adapter session."""
    used_adapter = STUDIO.has_session(req.session_id)
    if req.session_id and not used_adapter:
        raise HTTPException(status_code=404, detail=f"unknown session {req.session_id}")

    model = STUDIO.get_session(req.session_id) if used_adapter else STUDIO.fresh_model()
    prompt = STUDIO.formatter.prompt(req.instruction, req.input)
    prompt_ids = STUDIO.tokenizer.encode(prompt, add_eos=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    out_ids = model.generate(  # type: ignore[attr-defined]
        input_ids,
        max_new_tokens=req.max_new_tokens,
        eos_id=STUDIO.tokenizer.eos_id,
        temperature=req.temperature,
        greedy=req.greedy,
    )
    completion_ids = out_ids[0, len(prompt_ids):].tolist()
    completion = STUDIO.tokenizer.decode(completion_ids)

    return GenerateResponse(prompt=prompt, completion=completion, used_adapter=used_adapter)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
