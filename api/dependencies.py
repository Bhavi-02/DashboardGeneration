"""Shared FastAPI dependencies for Gen-Dash API routers"""
from typing import Dict, Any
from fastapi import Depends, Cookie, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

# Import database dependency
from core.database import get_db

# Import auth dependencies from auth module
from auth.auth import (
    get_session,
    create_session,
    delete_session,
    check_permission,
    require_auth,
    require_role,
    get_accessible_pages
)

# Import service layer dependencies
from services.dashboard_service import get_dashboard_system
from services.auth_service import get_password_hash, verify_password

# Re-export for easy importing in routers
__all__ = [
    'get_db',
    'get_session',
    'create_session',
    'delete_session',
    'check_permission',
    'require_auth',
    'require_role',
    'get_accessible_pages',
    'get_dashboard_system',
    'get_password_hash',
    'verify_password',
]

# Additional shared dependencies can be added here
def get_session_id(session_id: Optional[str] = Cookie(None)) -> Optional[str]:
    """Get session ID from cookie

    Args:
        session_id: Session ID from cookie

    Returns:
        Session ID or None
    """
    return session_id

def get_current_session(session_id: Optional[str] = Depends(get_session_id)) -> Optional[Dict[str, Any]]:
    """Get current session data

    Args:
        session_id: Session ID from dependency

    Returns:
        Session dictionary or None if not found
    """
    if session_id:
        return get_session(session_id)
    return None
