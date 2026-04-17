"""FastAPI service for Telco churn prediction with SHAP explainability.

Production-ready surface: ASGI lifespan loads ML artifacts once at startup,
a correlation-ID middleware stamps every log line with the request's
``x-request-id``, and global exception handlers return JSON errors in a
stable schema.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware

from api.schemas import CustomerInput
from src.explain import ChurnExplainer

# --------- Logging with request correlation ID -----------------------------
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def _configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    ))
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False


_configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("api")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = rid
            return response
        finally:
            request_id_ctx.reset(token)


# --------- Lifespan: load ML artifacts ONCE at startup ---------------------
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading ML artifacts from %s", MODELS_DIR)
    try:
        app.state.explainer = ChurnExplainer(
            model_path=str(MODELS_DIR / "pipeline.pkl"),
            background_path=str(MODELS_DIR / "background_data.pkl"),
            feature_names_path=str(MODELS_DIR / "feature_names.pkl"),
        )
        app.state.ready = True
        logger.info("ML artifacts loaded OK")
    except Exception:
        logger.exception("failed to load ML artifacts")
        app.state.explainer = None
        app.state.ready = False

    yield

    logger.info("releasing ML artifacts")
    app.state.explainer = None
    app.state.ready = False


# --------- App + middleware ------------------------------------------------
app = FastAPI(
    title="Telco Customer Churn API",
    description="Predicts customer churn probability with SHAP-based explanations",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------- Pydantic v2 response models -------------------------------------
class Factor(BaseModel):
    feature: str
    value: float
    impact: float = Field(..., description="SHAP value in logit space")
    direction: str


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: int
    probability: float
    expected_value: float
    top_factors: list[Factor]


class HealthResponse(BaseModel):
    status: str
    ready: bool


class RootResponse(BaseModel):
    service: str
    version: str
    docs: str


# --------- Dependencies ----------------------------------------------------
def get_explainer(request: Request) -> ChurnExplainer:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="model not ready")
    return request.app.state.explainer


# --------- Routes ----------------------------------------------------------
@app.get("/", response_model=RootResponse, tags=["meta"])
def root() -> RootResponse:
    return RootResponse(
        service="Telco Customer Churn API",
        version="1.0.0",
        docs="/docs",
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    ready = bool(getattr(request.app.state, "ready", False))
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "ready": False},
        )
    return HealthResponse(status="ok", ready=True)


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(
    customer: CustomerInput,
    explainer: ChurnExplainer = Depends(get_explainer),
) -> PredictResponse:
    payload: dict[str, Any] = customer.model_dump()
    result = explainer.predict_and_explain(payload, top_n=5)
    logger.info("predict ok prob=%.4f", result["probability"])
    return PredictResponse(**result)


# --------- Global error handlers -------------------------------------------
@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def on_http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception):
    logger.exception(
        "unhandled exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "internal server error"},
    )
