"""
Authentication and Authorization Module
Handles session management and role-based access control (RBAC)
"""
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import secrets
from fastapi import HTTPException, Cookie, Response
from pydantic import BaseModel

# Simple in-memory session store (in production, use Redis or database)
sessions: Dict[str, Dict] = {}

# Session timeout (30 minutes)
SESSION_TIMEOUT = timedelta(minutes=30)

# Role-based access control matrix
ROLE_PERMISSIONS = {
    "Admin": ["admin", "analyst", "departmental", "viewer"],
    "Analyst": ["analyst", "departmental", "viewer"],
    "Departmental": ["departmental", "viewer"],
    "Viewer": ["viewer"]
}

class Session(BaseModel):
    session_id: str
    user_id: int
    username: str
    email: str
    full_name: str
    role: str
    department: Optional[str]
    created_at: datetime
    last_accessed: datetime

def create_session(user_data: dict) -> str:
    """Create a new session for authenticated user"""
    session_id = secrets.token_urlsafe(32)
    session = {
        "session_id": session_id,
        "user_id": user_data["id"],
        "username": user_data["username"],
        "email": user_data["email"],
        "full_name": user_data["full_name"],
        "role": user_data["role"],
        "department": user_data.get("department"),
        "created_at": datetime.now(),
        "last_accessed": datetime.now()
    }
    sessions[session_id] = session
    return session_id

def get_session(session_id: Optional[str]) -> Optional[Dict]:
    """Get session data if valid"""
    if not session_id or session_id not in sessions:
        return None
    
    session = sessions[session_id]
    
    # Check if session has expired
    if datetime.now() - session["last_accessed"] > SESSION_TIMEOUT:
        del sessions[session_id]
        return None
    
    # Update last accessed time
    session["last_accessed"] = datetime.now()
    return session

def delete_session(session_id: Optional[str]) -> bool:
    """Delete a session (logout)"""
    if session_id and session_id in sessions:
        del sessions[session_id]
        return True
    return False

def check_permission(session: Dict, required_page: str) -> bool:
    """Check if user has permission to access a page"""
    user_role = session.get("role")
    if not user_role:
        return False
    
    allowed_pages = ROLE_PERMISSIONS.get(user_role, [])
    return required_page in allowed_pages

def get_accessible_pages(role: str) -> List[str]:
    """Get list of pages accessible to a role"""
    return ROLE_PERMISSIONS.get(role, [])

def require_auth(session_id: Optional[str] = Cookie(None)) -> Dict:
    """Dependency to require authentication"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please login first."
        )
    return session

def require_role(required_page: str):
    """Dependency to require specific role/permission"""
    def role_checker(session_id: Optional[str] = Cookie(None)) -> Dict:
        session = get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated. Please login first."
            )
        
        if not check_permission(session, required_page):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. You don't have permission to access {required_page} page."
            )
        
        return session
    return role_checker
