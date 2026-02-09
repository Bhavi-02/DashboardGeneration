"""Dashboard system service with singleton pattern"""
import sys
from pathlib import Path

# Add dashboard path to sys.path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

# Global dashboard system singleton (initialized lazily)
dashboard_system = None

def get_dashboard_system():
    """Get or initialize the dashboard system singleton.

    Returns:
        InteractiveDashboard instance or None if initialization fails

    Notes:
        - Dashboard system is initialized lazily on first use
        - Charts accumulate in dashboard_system.dashboard.charts[] across requests
        - Clear via dashboard_system.dashboard.clear_charts() before new sessions
        - Avoid multiple InteractiveDashboard() instances - use get_dashboard_system()
    """
    global dashboard_system

    if dashboard_system is None:
        try:
            from dashboard.interactive_dashboard import InteractiveDashboard
            dashboard_system = InteractiveDashboard()
            print("✅ Dashboard system initialized")
        except Exception as e:
            print(f"❌ Error initializing dashboard system: {e}")
            return None

    return dashboard_system
