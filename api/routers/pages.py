"""HTML page serving API router for Gen-Dash"""
from fastapi import APIRouter, Depends, Cookie, Response
from fastapi.responses import HTMLResponse, RedirectResponse, Response as FastAPIResponse
from pathlib import Path
from typing import Optional
from api.dependencies import require_auth, require_role, get_session, delete_session

router = APIRouter()

# Root redirect
@router.get("/", response_class=HTMLResponse)
async def read_root():
    """Redirect root to login page"""
    return RedirectResponse(url="/page.html")

@router.get("/page.html", response_class=HTMLResponse)
async def read_login():
    """Serve login page"""
    file_path = Path("Frontend/page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Login page not found</h1>", status_code=404)

@router.get("/register.html", response_class=HTMLResponse)
async def read_register():
    """Serve registration page"""
    file_path = Path("Frontend/register.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Register page not found</h1>", status_code=404)

@router.get("/style.css")
async def read_css():
    """Serve CSS stylesheet"""
    file_path = Path("Frontend/style.css")
    if file_path.exists():
        return FastAPIResponse(content=file_path.read_text(encoding='utf-8'), media_type="text/css")
    return FastAPIResponse(content="", media_type="text/css", status_code=404)

@router.get("/analytics_demo.html", response_class=HTMLResponse)
async def read_analytics_demo():
    """Serve analytics demo page"""
    file_path = Path("Frontend/analytics_demo.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Analytics Demo page not found</h1>", status_code=404)

@router.get("/user_guide.html", response_class=HTMLResponse)
async def read_user_guide():
    """Serve user guide page"""
    file_path = Path("Frontend/user_guide.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>User Guide page not found</h1>", status_code=404)

@router.get("/contact_support.html", response_class=HTMLResponse)
async def read_contact_support():
    """Serve contact support page"""
    file_path = Path("Frontend/contact_support.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Contact Support page not found</h1>", status_code=404)

@router.get("/dashboard-explainer", response_class=HTMLResponse)
async def dashboard_explainer_ui(session: dict = Depends(require_auth)):
    """Serve the dashboard explainability UI"""
    file_path = Path("Frontend/dashboard_explainer.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Dashboard Explainer not found</h1>", status_code=404)

@router.get("/home", response_class=HTMLResponse)
async def home_page(session_id: Optional[str] = Cookie(None)):
    """Home page (main dashboard selector based on role)"""
    session = get_session(session_id)
    if not session:
        return RedirectResponse(url="/page.html")

    role = session["role"]

    # Redirect to appropriate dashboard based on role
    if role == "Admin":
        file_path = Path("Frontend/admin_dashboard.html")
    elif role == "Analyst":
        file_path = Path("Frontend/analyst_dashboard.html")
    elif role == "Departmental":
        file_path = Path("Frontend/departmental_dashboard.html")
    elif role == "Viewer":
        file_path = Path("Frontend/viewer_dashboard.html")
    else:
        return RedirectResponse(url="/page.html")

    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

# Dashboard routes with RBAC
@router.get("/dashboard/admin", response_class=HTMLResponse)
async def admin_dashboard(session: dict = Depends(require_role("admin"))):
    """Serve admin dashboard page"""
    file_path = Path("Frontend/admin_page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Admin page not found</h1>", status_code=404)

@router.get("/dashboard/analyst", response_class=HTMLResponse)
async def analyst_dashboard(session: dict = Depends(require_role("analyst"))):
    """Serve analyst dashboard page"""
    file_path = Path("Frontend/analyst_page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Analyst page not found</h1>", status_code=404)

@router.get("/dashboard/departmental", response_class=HTMLResponse)
async def departmental_dashboard(session: dict = Depends(require_role("departmental"))):
    """Serve departmental dashboard page"""
    file_path = Path("Frontend/departmental_page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Departmental page not found</h1>", status_code=404)

@router.get("/dashboard/viewer", response_class=HTMLResponse)
async def viewer_dashboard(session: dict = Depends(require_role("viewer"))):
    """Serve viewer dashboard page"""
    file_path = Path("Frontend/viewer_page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Viewer page not found</h1>", status_code=404)

@router.post("/logout")
async def logout(response: Response, session_id: Optional[str] = Cookie(None)):
    """Logout endpoint"""
    if session_id:
        delete_session(session_id)
    response.delete_cookie("session_id")
    return {"message": "Logged out successfully"}

@router.get("/interactive-dashboard", response_class=HTMLResponse)
async def interactive_dashboard_ui(session: dict = Depends(require_auth)):
    """Serve the interactive dashboard builder UI"""
    file_path = Path("Frontend/interactive_builder.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Interactive dashboard builder not found</h1>", status_code=404)

@router.get("/chart-creator", response_class=HTMLResponse)
async def chart_creator_ui(session: dict = Depends(require_auth)):
    """Serve the simple chart creator UI"""
    file_path = Path("Frontend/chart_creator.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Chart creator not found</h1>", status_code=404)

@router.get("/smart-dashboard", response_class=HTMLResponse)
async def smart_dashboard_ui(session: dict = Depends(require_auth)):
    """Serve the AI-powered smart dashboard generator UI"""
    file_path = Path("Frontend/smart_dashboard.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Smart dashboard not found</h1>", status_code=404)
