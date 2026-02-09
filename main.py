"""
Gen-Dash: AI-Powered Dashboard Generation Platform
Version: 1.2.0 (Refactored for Production)
Last Updated: February 9, 2026

Main application entry point - Minimal and clean.
All business logic has been moved to routers and services.
"""

from fastapi import FastAPI

# Core configuration
from core.config import setup_logging, setup_cors
from core.database import Base, engine
from core.lifespan import lifespan

# API Routers
from api.routers import (
    auth,
    pages,
    datasets,
    charts,
    dashboards,
    ratings,
    exports,
    explainability,
    rag
)

# Setup logging
logger = setup_logging()
logger.info("🚀 Initializing Gen-Dash application...")

# Create database tables
Base.metadata.create_all(bind=engine)
logger.info("✅ Database tables initialized")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Gen-Dash",
    description="AI-Powered Dashboard Generation Platform",
    version="1.2.0",
    lifespan=lifespan
)

# Setup CORS middleware
setup_cors(app)
logger.info("✅ CORS middleware configured")

# Include routers
# Pages (HTML serving) - no prefix
app.include_router(pages.router, tags=["pages"])

# Authentication - no prefix for legacy compatibility
app.include_router(auth.router, tags=["auth"])

# API endpoints - with /api prefix
app.include_router(datasets.router, prefix="/api", tags=["datasets"])
app.include_router(charts.router, prefix="/api", tags=["charts"])
app.include_router(dashboards.router, prefix="/api", tags=["dashboards"])
app.include_router(ratings.router, prefix="/api", tags=["ratings"])
app.include_router(exports.router, prefix="/api", tags=["exports"])
app.include_router(explainability.router, prefix="/api", tags=["explainability"])
app.include_router(rag.router, prefix="/api", tags=["rag"])

logger.info("✅ All routers registered")
logger.info("""
╔══════════════════════════════════════════════════════════════╗
║                  Gen-Dash Server Ready                       ║
║                                                              ║
║  🌐 Server: http://localhost:8000                           ║
║  📚 API Docs: http://localhost:8000/docs                    ║
║  🔧 Admin: http://localhost:8000/home                       ║
║                                                              ║
║  Refactored Structure:                                       ║
║  ├── 3 Core modules (config, database, lifespan)           ║
║  ├── 4 Service modules (auth, feedback, dashboard, rating)  ║
║  └── 9 API routers (57 endpoints total)                     ║
╚══════════════════════════════════════════════════════════════╝
""")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": "1.2.0",
        "service": "Gen-Dash"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
