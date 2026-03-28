"""
CodePilot — FastAPI Application Entry Point
============================================
Responsibilities:
  - Define the application lifespan (startup / shutdown hooks)
  - Configure CORS middleware
  - Mount all API routers under their versioned prefix
  - Expose GET /health liveness + readiness probe
  - Initialise Sentry and OpenTelemetry instrumentation
  - Provide a Redis connection pool accessible via app.state.redis

Startup sequence:
  1. Sentry + OTEL instrumentation wired
  2. PostgreSQL async engine verified (init_db)
  3. Redis connection pool created and pinged
  4. Routers mounted

Shutdown sequence:
  1. Redis connection pool closed
  2. PostgreSQL engine disposed (close_db)
"""

from __future__ import annotations

import asyncio
import logging
import time
import datetime
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import sentry_sdk
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import ConnectionPool, Redis
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from backend.config import settings
from backend.db.engine import close_db, init_db

logger = logging.getLogger(__name__)

# ── Instrumentation setup (runs at import time — before app is created) ────────


def _setup_sentry() -> None:
    """Wire Sentry error tracking. Skipped if DSN is not configured."""
    dsn = settings.observability.sentry_dsn
    if not dsn:
        logger.info("SENTRY_DSN not set — Sentry disabled")
        return

    sentry_sdk.init(
        dsn=dsn,
        environment=settings.environment,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
        ],
        # Only sample 100% of transactions in dev; tune for prod
        traces_sample_rate=1.0 if settings.is_development else 0.2,
        send_default_pii=False,
    )
    logger.info("Sentry initialised (env=%s)", settings.environment)


def _setup_otel() -> None:
    """
    Wire OpenTelemetry tracing if an OTLP endpoint is configured.
    Instruments FastAPI (HTTP spans) and SQLAlchemy (query spans).
    """
    endpoint = settings.observability.otlp_endpoint
    if not endpoint:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — OTel disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource(
            attributes={SERVICE_NAME: settings.observability.otel_service_name}
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()

        logger.info(
            "OpenTelemetry initialised (endpoint=%s service=%s)",
            endpoint,
            settings.observability.otel_service_name,
        )
    except ImportError as exc:
        logger.warning("OTel packages not installed — tracing disabled: %s", exc)


_setup_sentry()
_setup_otel()

# ── Redis pool factory ─────────────────────────────────────────────────────────


def _create_redis_pool() -> ConnectionPool:
    redis_cfg = settings.redis
    return ConnectionPool.from_url(
        redis_cfg.url,
        max_connections=redis_cfg.max_connections,
        socket_timeout=redis_cfg.socket_timeout,
        socket_connect_timeout=redis_cfg.socket_timeout,
        decode_responses=False,  # raw bytes — callers decode as needed
        health_check_interval=30,
    )


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup; after `yield` runs at shutdown.
    """
    # ── STARTUP ────────────────────────────────────────────────────────────────
    logger.info("CodePilot backend starting up (env=%s)", settings.environment)

    # 1. PostgreSQL — verify connection and pgvector availability
    logger.info("Connecting to PostgreSQL...")
    await init_db()

    # 2. Redis — create pool and verify connectivity
    logger.info("Connecting to Redis...")
    redis_pool = _create_redis_pool()
    redis_client: Redis = Redis(connection_pool=redis_pool)
    try:
        await redis_client.ping()
        logger.info("Redis connection verified.")
    except Exception as exc:
        logger.error("Redis connection failed: %s", exc)
        raise

    # Expose shared resources on app.state for dependency injection
    app.state.redis = redis_client
    app.state.redis_pool = redis_pool

    # 3. Pipeline worker — background asyncio task consuming the Redis pipeline stream
    logger.info("Starting pipeline worker background task...")
    from backend.worker import run_worker
    worker_task = asyncio.create_task(
        run_worker(redis_client=redis_client),
        name="pipeline-worker",
    )
    app.state.worker_task = worker_task

    logger.info("CodePilot backend ready.")

    yield  # ── Application serves requests here ──

    # ── SHUTDOWN ───────────────────────────────────────────────────────────────
    logger.info("CodePilot backend shutting down...")

    # Cancel and await the pipeline worker
    worker_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(worker_task), timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    logger.info("Pipeline worker stopped.")

    # Close Redis pool gracefully
    await redis_client.aclose()
    await redis_pool.aclose()
    logger.info("Redis pool closed.")

    # Dispose SQLAlchemy engine / connection pool
    await close_db()

    logger.info("Shutdown complete.")


# ── Application factory ────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application.
    Separated into a factory function for testability.
    """
    app = FastAPI(
        title="CodePilot API",
        description=(
            "Autonomous coding agent SaaS — Linear → GitHub PR pipeline. "
            "Powers ticket enrichment, difficulty scoring, multi-agent wave execution, "
            "security scanning, and codebase memory."
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    # allow_origins=["*"] + allow_credentials=True is rejected by browsers.
    # Use allow_origins_regex to match any origin while supporting credentials.
    import os as _os
    _extra_origins = [
        o.strip() for o in _os.getenv("EXTRA_CORS_ORIGINS", "").split(",") if o.strip()
    ]
    allowed_origins = list({
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        settings.frontend_url,
        *_extra_origins,
    })

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"https?://.*",  # allow any http/https origin (dev convenience)
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Run-ID"],
    )

    # ── Request ID middleware ──────────────────────────────────────────────────
    # Injects X-Request-ID into every response for distributed tracing correlation.
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Any:
        import uuid as _uuid

        request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Routers ────────────────────────────────────────────────────────────────
    # Routers are imported lazily inside this function to:
    #   1. Avoid circular imports at module level
    #   2. Allow test fixtures to monkeypatch before routers are registered
    #
    # Each router is defined in its own module under backend/routers/.
    # Imports are guarded with a try/except so the app can boot even if an
    # individual router module has a dependency issue during development.

    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """
    Mount all API routers. Called once from create_app().

    Router modules that don\'t exist yet are skipped with a warning so
    the development server can boot incrementally as files are added.
    """
    _api_prefix = f"/api/{settings.api_version}"

    router_specs = [
        # (import path, attribute name, prefix, tags)
        ("backend.routers.runs", "router", f"{_api_prefix}/runs", ["runs"]),
        ("backend.routers.tickets", "router", f"{_api_prefix}/tickets", ["tickets"]),
        ("backend.routers.insights", "router", f"{_api_prefix}/insights", ["insights"]),
        ("backend.routers.org", "router", f"{_api_prefix}/org", ["org"]),
        ("backend.routers.org", "public_router", f"{_api_prefix}/public", ["public"]),
        ("backend.routers.billing", "router", f"{_api_prefix}/billing", ["billing"]),
        ("backend.routers.auth", "router", f"{_api_prefix}/auth", ["auth"]),
        ("backend.routers.webhooks.linear", "router", "/webhooks/linear", ["webhooks"]),
        ("backend.routers.webhooks.github", "router", "/webhooks/github", ["webhooks"]),
        ("backend.routers.webhooks.slack", "router", "/webhooks/slack", ["webhooks"]),
    ]

    for module_path, attr, prefix, tags in router_specs:
        try:
            import importlib
            module = importlib.import_module(module_path)
            router = getattr(module, attr)
            app.include_router(router, prefix=prefix, tags=tags)
            logger.debug("Router mounted: %s -> %s", module_path, prefix)
        except ModuleNotFoundError:
            logger.warning(
                "Router module not found (will be skipped): %s "
                "— create the file to activate this router.",
                module_path,
            )
        except AttributeError:
            logger.warning(
                "Router module %s has no attribute %r — skipping.",
                module_path,
                attr,
            )


# ── Application instance ───────────────────────────────────────────────────────

app = create_app()


# ── Health endpoint ────────────────────────────────────────────────────────────


@app.get(
    "/health",
    tags=["health"],
    summary="Liveness + readiness probe",
    response_description="Service health status with DB, Redis connectivity, and timestamp",
)
async def health_check(request: Request) -> JSONResponse:
    """
    Combined liveness and readiness probe.

    Returns HTTP 200 if all downstream dependencies are reachable.
    Returns HTTP 503 if any dependency is unhealthy.

    Used by:
      - Docker HEALTHCHECK
      - Kubernetes liveness + readiness probes
      - Load balancer health checks
    """
    start = time.monotonic()
    checks: dict[str, Any] = {}
    overall_ok = True

    # ── PostgreSQL check ───────────────────────────────────────────────────────
    try:
        from sqlalchemy import text
        from backend.db.engine import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception as exc:
        checks["database"] = {"status": "error", "detail": str(exc)}
        overall_ok = False
        logger.error("Health check: database error: %s", exc)

    # ── Redis check ────────────────────────────────────────────────────────────
    try:
        redis: Redis = request.app.state.redis
        await redis.ping()
        checks["redis"] = {"status": "ok"}
    except Exception as exc:
        checks["redis"] = {"status": "error", "detail": str(exc)}
        overall_ok = False
        logger.error("Health check: redis error: %s", exc)

    # ── Response ───────────────────────────────────────────────────────────────
    elapsed_ms = round((time.monotonic() - start) * 1000, 2)
    checked_at = datetime.datetime.utcnow().isoformat() + 'Z'
    body = {
        "status": "ok" if overall_ok else "degraded",
        "checked_at": checked_at,
        "environment": settings.environment,
        "version": app.version,
        "checks": checks,
        "latency_ms": elapsed_ms,
    }

    http_status = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=body, status_code=http_status)


# ── Global exception handler ───────────────────────────────────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unhandled exceptions. Returns a structured JSON error
    instead of a raw 500 HTML page. Sentry will capture the exception
    automatically via its FastAPI integration.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. The incident has been logged.",
            "path": str(request.url.path),
        },
    )


# ── Uvicorn entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.observability.log_level.lower(),
        # Production: use gunicorn + UvicornWorker instead
        # gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker
    )
