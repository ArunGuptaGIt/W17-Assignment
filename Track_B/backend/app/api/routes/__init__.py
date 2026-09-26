from app.api.routes.chat import router as chat_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.compare import router as compare_router
from app.api.routes.health import router as health_router

__all__ = ["chat_router", "ingest_router", "compare_router", "health_router"]
