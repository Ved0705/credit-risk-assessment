"""FastAPI entry point.  Run:  uvicorn app.main:app --reload  (from backend/)"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .database.store import get_repository
from .ml.model_loader import ModelNotFoundError, get_model
from .routes import fairness, model, prediction
from .schemas.responses import HealthResponse

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        get_model()
        log.info("Model loaded")
    except ModelNotFoundError as exc:
        log.error(str(exc))
    get_repository()
    yield


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    description="Explainable credit-risk prediction on the German Credit dataset, with SHAP "
                "explanations, a Fairlearn fairness audit and a stability test. "
                "Educational demo, not a real lending system.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origin_list,
    # any local dev port (Vite moves to 5174, 5175… if 5173 is busy)
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


@app.exception_handler(ModelNotFoundError)
async def model_missing(_: Request, exc: ModelNotFoundError):
    return JSONResponse(status_code=503, content={"error": "model_not_available", "detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def invalid_input(_: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": errors})


@app.exception_handler(Exception)
async def unexpected(_: Request, exc: Exception):
    log.exception("Unhandled error")  # full trace in server log only
    return JSONResponse(status_code=500, content={"error": "internal_error",
                                                  "detail": "The request could not be processed."})



@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    repo = get_repository()
    try:
        b = get_model()
        return {"status": "ok", "model_loaded": True, "model_name": b.name, "database": repo.backend}
    except ModelNotFoundError as exc:
        return {"status": "degraded", "model_loaded": False, "database": repo.backend, "detail": str(exc)}


app.include_router(prediction.router)
app.include_router(model.router)
app.include_router(fairness.router)
