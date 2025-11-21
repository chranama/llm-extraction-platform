# app/services/llm_api.py
from __future__ import annotations

from typing import Iterator, Optional, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.llm import ModelManager, DEFAULT_STOPS

app = FastAPI(title="LLM Runtime (host)", version="0.1.0")
_llm = ModelManager()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    output: str
    model: str


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    try:
        _llm.ensure_loaded()
        return {"status": "ready", "model": _llm.model_id}
    except Exception:
        return {"status": "not ready"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest):
    stops = body.stop if (body.stop and len(body.stop) > 0) else DEFAULT_STOPS
    _llm.ensure_loaded()
    out = _llm.generate(
        prompt=body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        stop=stops,
    )
    return GenerateResponse(output=out, model=_llm.model_id)


@app.post("/stream")
def stream(body: GenerateRequest) -> StreamingResponse:
    stops = body.stop if (body.stop and len(body.stop) > 0) else DEFAULT_STOPS

    def sse_lines() -> Iterator[bytes]:
        try:
            for chunk in _llm.stream(
                prompt=body.prompt,
                max_new_tokens=body.max_new_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                top_k=body.top_k,
                repetition_penalty=body.repetition_penalty,
                stop=stops,
            ):
                yield f"data: {chunk}\n\n".encode("utf-8")
        finally:
            yield b"data: [DONE]\n\n"

    return StreamingResponse(sse_lines(), media_type="text/event-stream")