"""FastAPI lifespan event handler for startup and shutdown"""
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Import here to avoid circular dependencies
    from services.dashboard_service import dashboard_system

    # Startup
    print("🚀 Server starting up...")
    # Dashboard system will be initialized on first use, ensuring clean state

    yield

    # Shutdown
    if dashboard_system is not None:
        try:
            dashboard_system.dashboard.clear_charts()
            print("🧹 Dashboard charts cleared on shutdown")
        except Exception as e:
            print(f"⚠️ Error clearing charts on shutdown: {e}")
