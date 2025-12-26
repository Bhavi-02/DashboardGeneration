from fastapi import FastAPI, HTTPException, Form, Request, Response, Cookie, Depends, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
import bcrypt
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager
import sys
import os
import pandas as pd
import zipfile
from datetime import datetime
import plotly.io as pio
import json
import io
import traceback
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/gendash.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add dashboard path to sys.path
dashboard_path = Path(__file__).parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from database.models import Base, User, Dashboard
from database.schemas import UserCreate, UserLogin, UserResponse
from auth.auth import (
    create_session, get_session, delete_session, 
    check_permission, require_auth, require_role,
    get_accessible_pages
)

# Import dashboard components
try:
    from dashboard.interactive_dashboard import InteractiveDashboard
    dashboard_system = None  # Will be initialized on first use
except ImportError as e:
    print(f"Warning: Could not import dashboard system: {e}")
    dashboard_system = None

# Import dashboard explainer
try:
    from dashboard_explainer import get_dashboard_explainer
    print("✅ Dashboard Explainer loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import dashboard explainer: {e}")
    get_dashboard_explainer = None

DATABASE_URL = "mysql+pymysql://root:dhruv123@localhost:3306/analytics_dashboard"

# Create engine with connection pooling and optimizations
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_size=5,         # Connection pool size
    max_overflow=10,     # Maximum overflow connections
    echo=False           # Set to True for SQL debugging
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database dependency for automatic session management
def get_db():
    """Dependency to get database session with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global dashboard_system
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

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# Routes to serve HTML pages
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return RedirectResponse(url="/page.html")

@app.get("/page.html", response_class=HTMLResponse)
async def read_login():
    file_path = Path("Frontend/page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Login page not found</h1>", status_code=404)

@app.get("/register.html", response_class=HTMLResponse)
async def read_register():
    file_path = Path("Frontend/register.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Register page not found</h1>", status_code=404)

@app.get("/style.css")
async def read_css():
    file_path = Path("Frontend/style.css")
    if file_path.exists():
        from fastapi.responses import Response
        return Response(content=file_path.read_text(encoding='utf-8'), media_type="text/css")
    return Response(content="", media_type="text/css", status_code=404)

@app.get("/analytics_demo.html", response_class=HTMLResponse)
async def read_analytics_demo():
    file_path = Path("Frontend/analytics_demo.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Analytics Demo page not found</h1>", status_code=404)

@app.get("/user_guide.html", response_class=HTMLResponse)
async def read_user_guide():
    file_path = Path("Frontend/user_guide.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>User Guide page not found</h1>", status_code=404)

@app.get("/contact_support.html", response_class=HTMLResponse)
async def read_contact_support():
    file_path = Path("Frontend/contact_support.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Contact Support page not found</h1>", status_code=404)

@app.get("/dashboard-explainer", response_class=HTMLResponse)
async def dashboard_explainer_ui(session: dict = Depends(require_auth)):
    """Serve the dashboard explainability UI"""
    file_path = Path("Frontend/dashboard_explainer.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Dashboard Explainer not found</h1>", status_code=404)

# New endpoint to get session info
@app.get("/api/session-info")
async def get_session_info(session: dict = Depends(require_auth)):
    """Return current user's session information"""
    return JSONResponse({
        "user_id": session["user_id"],
        "username": session["username"],
        "email": session["email"],
        "full_name": session["full_name"],
        "role": session["role"],
        "department": session.get("department")
    })

# Check session endpoint (for frontend validation)
@app.get("/api/check-session")
async def check_session(session_id: Optional[str] = Cookie(None)):
    """Check if user has a valid session"""
    session = get_session(session_id)
    if session:
        return JSONResponse({
            "authenticated": True,
            "user": session["username"],
            "role": session["role"]
        })
    return JSONResponse({
        "authenticated": False
    })

# Home page (main dashboard selector based on role)
@app.get("/home", response_class=HTMLResponse)
async def home_page(session_id: Optional[str] = Cookie(None)):
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
@app.get("/dashboard/admin", response_class=HTMLResponse)
async def admin_dashboard(session: dict = Depends(require_role("admin"))):
    file_path = Path("Frontend/admin_page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Admin page not found</h1>", status_code=404)

@app.get("/dashboard/analyst", response_class=HTMLResponse)
async def analyst_dashboard(session: dict = Depends(require_role("analyst"))):
    file_path = Path("Frontend/analyst_page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Analyst page not found</h1>", status_code=404)

@app.get("/dashboard/departmental", response_class=HTMLResponse)
async def departmental_dashboard(session: dict = Depends(require_role("departmental"))):
    file_path = Path("Frontend/departmental_page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Departmental page not found</h1>", status_code=404)

@app.get("/dashboard/viewer", response_class=HTMLResponse)
async def viewer_dashboard(session: dict = Depends(require_role("viewer"))):
    file_path = Path("Frontend/viewer_page.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Viewer page not found</h1>", status_code=404)

# Logout endpoint
@app.post("/logout")
async def logout(response: Response, session_id: Optional[str] = Cookie(None)):
    if session_id:
        delete_session(session_id)
    response.delete_cookie("session_id")
    return {"message": "Logged out successfully"}

# Interactive Dashboard Builder UI
@app.get("/interactive-dashboard", response_class=HTMLResponse)
async def interactive_dashboard_ui(session: dict = Depends(require_auth)):
    """Serve the interactive dashboard builder UI"""
    file_path = Path("Frontend/interactive_builder.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Interactive dashboard builder not found</h1>", status_code=404)

# Chart Creator UI
@app.get("/chart-creator", response_class=HTMLResponse)
async def chart_creator_ui(session: dict = Depends(require_auth)):
    """Serve the simple chart creator UI"""
    file_path = Path("Frontend/chart_creator.html")
    if file_path.exists():
        return HTMLResponse(content=file_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Chart creator not found</h1>", status_code=404)

# API: Get dataset information
@app.get("/api/get-dataset-info")
async def get_dataset_info(session: dict = Depends(require_auth)):
    """Return dataset schema information (tables, columns, types, row counts)"""
    try:
        print("📋 Fetching dataset information...")
        
        # Get or initialize dashboard system
        ds = get_dashboard_system()
        if not ds:
            print("❌ Dashboard system not available")
            return JSONResponse({
                "success": False,
                "error": "Dashboard system not available"
            }, status_code=500)
        
        print("✅ Dashboard system loaded")
        
        # Access the chart generator's data connector
        if not ds.nlu_pipeline:
            print("❌ NLU pipeline not available")
            return JSONResponse({
                "success": False,
                "error": "NLU pipeline not available"
            }, status_code=500)
            
        if not ds.nlu_pipeline.chart_generator:
            print("❌ Chart generator not available")
            return JSONResponse({
                "success": False,
                "error": "Chart generator not available"
            }, status_code=500)
        
        print("✅ Chart generator loaded")
        
        data_connector = ds.nlu_pipeline.chart_generator.data_connector
        
        if not data_connector:
            print("❌ Data connector not available")
            return JSONResponse({
                "success": False,
                "error": "Data connector not available"
            }, status_code=500)
        
        print(f"✅ Data connector loaded with {len(data_connector.cached_data)} tables")
        
        # Get current dataset info
        current_dataset = data_connector.get_current_dataset()
        all_datasets = data_connector.get_available_datasets()
        
        # Get all columns info for current dataset
        columns_info = data_connector.extract_all_columns_info()
        
        # Format for frontend display
        dataset_info = {
            "current_dataset": current_dataset,
            "available_datasets": all_datasets,
            "tables": []
        }
        
        for table_name, info in columns_info.items():
            table_data = {
                "name": table_name,
                "display_name": table_name.replace('_', ' ').title(),
                "row_count": info['row_count'],
                "all_columns": info['all_columns'],
                "numeric_columns": info['numeric_columns'],
                "text_columns": info['text_columns'],
                "date_columns": info['date_columns'],
                "sample_data": []  # Temporarily disable sample data to fix datetime issue
            }
            
            dataset_info["tables"].append(table_data)
        
        return JSONResponse({
            "success": True,
            "dataset_info": dataset_info
        })
        
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error retrieving dataset information: {str(e)}"
        }, status_code=500)

# API: Get all available datasets
@app.get("/api/get-available-datasets")
async def get_available_datasets(session: dict = Depends(require_auth)):
    """Return list of all available datasets"""
    try:
        ds = get_dashboard_system()
        if not ds or not ds.nlu_pipeline or not ds.nlu_pipeline.chart_generator:
            return JSONResponse({
                "success": False,
                "error": "Dashboard system not available"
            }, status_code=500)
        
        data_connector = ds.nlu_pipeline.chart_generator.data_connector
        datasets_info = data_connector.get_dataset_info_all()
        
        return JSONResponse({
            "success": True,
            "datasets": datasets_info,
            "current_dataset": data_connector.get_current_dataset()
        })
        
    except Exception as e:
        print(f"Error getting available datasets: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# API: Switch to a different dataset
@app.post("/api/switch-dataset")
async def switch_dataset(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Switch the active dataset for the current user session"""
    try:
        dataset_name = request_data.get('dataset_name')
        
        if not dataset_name:
            return JSONResponse({
                "success": False,
                "message": "Dataset name is required"
            }, status_code=400)
        
        ds = get_dashboard_system()
        if not ds or not ds.nlu_pipeline or not ds.nlu_pipeline.chart_generator:
            return JSONResponse({
                "success": False,
                "message": "Dashboard system not available"
            }, status_code=500)
        
        data_connector = ds.nlu_pipeline.chart_generator.data_connector
        
        # Switch the dataset
        success = data_connector.switch_dataset(dataset_name)
        
        if success:
            # Clear existing charts when switching datasets
            ds.dashboard.clear_charts()
            
            return JSONResponse({
                "success": True,
                "message": f"Switched to dataset: {dataset_name}",
                "current_dataset": dataset_name
            })
        else:
            return JSONResponse({
                "success": False,
                "message": f"Dataset '{dataset_name}' not found"
            }, status_code=404)
        
    except Exception as e:
        print(f"Error switching dataset: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)

# Initialize dashboard system
def get_dashboard_system():
    """Get or initialize the dashboard system"""
    global dashboard_system
    if dashboard_system is None:
        try:
            dashboard_system = InteractiveDashboard()
        except Exception as e:
            print(f"Error initializing dashboard system: {e}")
            return None
    return dashboard_system

# API: Add chart from query
@app.post("/api/add-chart")
async def add_chart(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Add a chart from a natural language query"""
    try:
        query = request_data.get('query', '')
        if query:
            query = query.strip()
        
        if not query:
            return JSONResponse({
                "success": False,
                "message": "Query cannot be empty"
            }, status_code=400)
        
        # Get dashboard system
        ds = get_dashboard_system()
        if not ds:
            return JSONResponse({
                "success": False,
                "message": "Dashboard system not available. Make sure the NLU model is trained."
            }, status_code=500)
        
        # Safety check: If chart count is suspiciously high (>20), auto-clear
        # This prevents accumulation if user forgot to generate dashboard
        current_count = ds.dashboard.get_chart_count()
        if current_count > 20:
            print(f"⚠️ Chart count ({current_count}) exceeded limit. Auto-clearing old charts.")
            ds.dashboard.clear_charts()
        
        # Add chart
        success = ds.add_chart_from_query(query)
        
        if success:
            chart_count = ds.dashboard.get_chart_count()
            return JSONResponse({
                "success": True,
                "message": f"Chart added successfully! Total charts: {chart_count}",
                "chart_count": chart_count
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "Failed to generate chart from query. Please check your query format."
            }, status_code=400)
            
    except Exception as e:
        print(f"Error adding chart: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error processing query: {str(e)}"
        }, status_code=500)

# API: Delete chart
@app.post("/api/delete-chart")
async def delete_chart(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Delete a chart by ID"""
    try:
        chart_id = request_data.get('chart_id')
        
        ds = get_dashboard_system()
        if not ds or not ds.dashboard:
            return JSONResponse({
                "success": False,
                "message": "Dashboard system not available"
            }, status_code=500)
        
        # For now, just acknowledge the delete
        # The actual chart deletion would need to be implemented in dashboard_generator.py
        return JSONResponse({
            "success": True,
            "message": "Chart deleted"
        })
            
    except Exception as e:
        print(f"Error deleting chart: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Error deleting chart: {str(e)}"
        }, status_code=500)

# API: Clear all charts
@app.post("/api/clear-charts")
async def clear_charts(session: dict = Depends(require_auth)):
    """Clear all charts"""
    try:
        ds = get_dashboard_system()
        if not ds or not ds.dashboard:
            return JSONResponse({
                "success": False,
                "message": "Dashboard system not available"
            }, status_code=500)
        
        # Clear all charts
        ds.dashboard.clear_charts()
        
        return JSONResponse({
            "success": True,
            "message": "All charts cleared"
        })
            
    except Exception as e:
        print(f"Error clearing charts: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Error clearing charts: {str(e)}"
        }, status_code=500)

# API: Generate dashboard
@app.post("/api/generate-dashboard")
async def generate_dashboard_api(session: dict = Depends(require_auth)):
    """Generate the final dashboard with all charts"""
    try:
        ds = get_dashboard_system()
        if not ds or not ds.dashboard:
            return JSONResponse({
                "success": False,
                "message": "Dashboard system not available"
            }, status_code=500)
        
        chart_count = ds.dashboard.get_chart_count()
        
        if chart_count == 0:
            return JSONResponse({
                "success": False,
                "message": "No charts to generate. Please add at least one chart."
            }, status_code=400)
        
        # Generate dashboard
        fig = ds.generate_and_save_dashboard(
            filename="interactive_dashboard.html",
            title=f"Analytics Dashboard - {session['full_name']}"
        )
        
        if fig:
            # Create a snapshot copy immediately after generation
            # This prevents the file from being overwritten by the next dashboard generation
            import shutil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"dashboard_snapshot_{session['user_id']}_{timestamp}.html"
            snapshot_path = Path("temp_dashboards") / snapshot_filename
            
            # Create temp directory if it doesn't exist
            snapshot_path.parent.mkdir(exist_ok=True)
            
            # Copy the generated dashboard to snapshot
            source_file = Path("interactive_dashboard.html")
            if source_file.exists():
                shutil.copy(source_file, snapshot_path)
                print(f"📸 Dashboard snapshot created: {snapshot_path}")
            
            # Clear charts after successful generation to prevent accumulation
            # This ensures the next dashboard only contains newly added charts
            ds.dashboard.clear_charts()
            print(f"✅ Dashboard generated and charts cleared for next session")
            
            # Add timestamp to prevent browser caching
            import time
            cache_timestamp = int(time.time() * 1000)
            return JSONResponse({
                "success": True,
                "message": f"Dashboard generated with {chart_count} charts",
                "chart_count": chart_count,
                "dashboard_url": f"/view-dashboard?v={cache_timestamp}",
                "snapshot_file": str(snapshot_path)  # Return snapshot path for saving
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "Failed to generate dashboard"
            }, status_code=500)
            
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error generating dashboard: {str(e)}"
        }, status_code=500)

# View generated dashboard
@app.get("/view-dashboard", response_class=HTMLResponse)
async def view_dashboard(session: dict = Depends(require_auth)):
    """View the latest generated dashboard"""
    file_path = Path("interactive_dashboard.html")
    if file_path.exists():
        # Prevent browser caching by adding cache-control headers
        return FileResponse(
            file_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    else:
        return HTMLResponse(
            content="<h1>Dashboard not found</h1><p>Please generate a dashboard first.</p>",
            status_code=404
        )

@app.get("/view-saved-dashboard/{dashboard_id}", response_class=HTMLResponse)
async def view_saved_dashboard(dashboard_id: int, session: dict = Depends(require_auth)):
    """View a specific saved dashboard by ID with RBAC checks"""
    db = SessionLocal()
    try:
        dashboard = db.query(Dashboard).filter(Dashboard.id == dashboard_id).first()
        
        if not dashboard:
            return HTMLResponse(
                content="<h1>Dashboard not found</h1><p>This dashboard does not exist.</p>",
                status_code=404
            )
        
        # RBAC: Check if user has access to view this dashboard
        user_role = session.get('role', '').lower()
        user_department = session.get('department')
        user_id = session['user_id']
        
        has_access = False
        
        # Owner always has access
        if dashboard.user_id == user_id:
            has_access = True
        # Viewer role: only if marked as visible_to_viewer
        elif user_role == 'viewer' and dashboard.visible_to_viewer:
            has_access = True
        # Departmental role: only if dashboard was created by another Departmental user and their department is in allowed_departments
        elif user_role == 'departmental' and dashboard.created_by_role == 'Departmental' and dashboard.allowed_departments and user_department:
            allowed_depts = [d.strip() for d in dashboard.allowed_departments.split(',')]
            if user_department in allowed_depts:
                has_access = True
        # Admin can see all dashboards
        elif user_role == 'admin':
            has_access = True
        
        if not has_access:
            return HTMLResponse(
                content="<h1>Access Denied</h1><p>You don't have permission to view this dashboard.</p>",
                status_code=403
            )
        
        # Check if dashboard file exists
        file_path = Path(dashboard.file_path)
        if file_path.exists():
            # Return the saved dashboard with cache-busting headers
            return FileResponse(
                file_path,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        else:
            # If file doesn't exist but we have config, recreate it
            return HTMLResponse(
                content=f"<h1>Dashboard File Missing</h1><p>The dashboard file '{dashboard.file_path}' was not found. Please regenerate the dashboard.</p>",
                status_code=404
            )
    except Exception as e:
        print(f"Error viewing saved dashboard: {str(e)}")
        return HTMLResponse(
            content=f"<h1>Error</h1><p>An error occurred: {str(e)}</p>",
            status_code=500
        )
    finally:
        db.close()

# API: Save dashboard to database
@app.post("/api/save-dashboard") 
async def save_dashboard(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Save dashboard configuration to database"""
    db = SessionLocal()
    try:
        title = request_data.get('title', '').strip()
        description = request_data.get('description', '').strip()
        charts_config = request_data.get('charts_config', [])
        snapshot_file = request_data.get('snapshot_file')  # Get snapshot file from frontend
        source_file = request_data.get('file_path', 'interactive_dashboard.html')
        
        if not title:
            return JSONResponse({
                "success": False,
                "message": "Dashboard title is required"
            }, status_code=400)
        
        if not charts_config or len(charts_config) == 0:
            return JSONResponse({
                "success": False,
                "message": "Cannot save empty dashboard. Add at least one chart."
            }, status_code=400)
        
        # Create dashboards directory if it doesn't exist
        dashboards_dir = Path("saved_dashboards")
        dashboards_dir.mkdir(exist_ok=True)
        
        # Generate unique filename for this dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
        unique_filename = f"dashboard_{session['user_id']}_{safe_title}_{timestamp}.html"
        unique_file_path = dashboards_dir / unique_filename
        
        # Determine source file: prefer snapshot if available, otherwise use main file
        if snapshot_file:
            source_path = Path(snapshot_file)
            print(f"📸 Using snapshot file: {source_path}")
        else:
            source_path = Path(source_file)
            print(f"⚠️ No snapshot file, using main file: {source_path}")
        
        # Copy the dashboard file to the unique location
        if source_path.exists():
            import shutil
            shutil.copy(source_path, unique_file_path)
            print(f"✅ Dashboard copied: {source_path} -> {unique_file_path}")
        else:
            return JSONResponse({
                "success": False,
                "message": f"Source dashboard file not found: {source_path}. Please generate the dashboard first."
            }, status_code=400)
        
        # Convert charts config to JSON string
        import json
        charts_json = json.dumps(charts_config)
        
        # Get visibility settings from request
        visible_to_viewer = request_data.get('visible_to_viewer', False)
        allowed_departments = request_data.get('allowed_departments', [])
        
        # Debug logging
        print(f"🔐 RBAC Settings Received:")
        print(f"   - visible_to_viewer: {visible_to_viewer} (type: {type(visible_to_viewer)})")
        print(f"   - allowed_departments: {allowed_departments}")
        print(f"   - user_role: {session['role']}")
        
        # Convert allowed_departments list to comma-separated string
        allowed_departments_str = ','.join(allowed_departments) if allowed_departments else None
        
        # Create new dashboard record with unique file path and RBAC settings
        dashboard = Dashboard(
            user_id=session['user_id'],
            title=title,
            description=description,
            charts_config=charts_json,
            file_path=str(unique_file_path),
            chart_count=len(charts_config),
            created_by_role=session['role'],  # Store creator's role
            visible_to_viewer=visible_to_viewer,
            allowed_departments=allowed_departments_str
        )
        
        db.add(dashboard)
        db.commit()
        db.refresh(dashboard)
        
        print(f"✅ Dashboard saved: {title} -> {unique_file_path}")
        
        return JSONResponse({
            "success": True,
            "message": f"Dashboard '{title}' saved successfully!",
            "dashboard_id": dashboard.id
        })
        
    except Exception as e:
        print(f"Error saving dashboard: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error saving dashboard: {str(e)}"
        }, status_code=500)
    finally:
        db.close()

# API: Get current session chart count (for debugging/monitoring)
@app.get("/api/current-chart-count")
async def get_current_chart_count(session: dict = Depends(require_auth)):
    """Get the current number of charts in the session (for monitoring)"""
    try:
        ds = get_dashboard_system()
        if not ds or not ds.dashboard:
            return JSONResponse({
                "success": True,
                "chart_count": 0,
                "message": "Dashboard system not initialized"
            })
        
        chart_count = ds.dashboard.get_chart_count()
        return JSONResponse({
            "success": True,
            "chart_count": chart_count,
            "message": f"Current session has {chart_count} charts",
            "warning": "High chart count - consider generating dashboard" if chart_count > 15 else None
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# API: Debug RBAC - Check what dashboards are visible (for testing)
@app.get("/api/debug-rbac")
async def debug_rbac(session: dict = Depends(require_auth)):
    """Debug endpoint to check RBAC visibility logic"""
    db = SessionLocal()
    try:
        user_role = session['role']
        user_department = session.get('department')
        user_id = session['user_id']
        
        # Get ALL dashboards
        all_dashboards = db.query(Dashboard).filter(Dashboard.is_active == True).all()
        
        debug_info = {
            "current_user": {
                "id": user_id,
                "role": user_role,
                "department": user_department
            },
            "total_dashboards": len(all_dashboards),
            "dashboards": []
        }
        
        for d in all_dashboards:
            dashboard_info = {
                "id": d.id,
                "title": d.title,
                "user_id": d.user_id,
                "created_by_role": d.created_by_role,
                "visible_to_viewer": d.visible_to_viewer,
                "allowed_departments": d.allowed_departments,
                "is_owner": d.user_id == user_id,
                "visible_to_current_user": False,
                "reason": ""
            }
            
            # Check if visible to current user
            if d.user_id == user_id:
                dashboard_info["visible_to_current_user"] = True
                dashboard_info["reason"] = "Owner"
            elif user_role.lower() == 'admin':
                dashboard_info["visible_to_current_user"] = True
                dashboard_info["reason"] = "Admin - full access"
            elif user_role.lower() == 'viewer' and d.visible_to_viewer:
                dashboard_info["visible_to_current_user"] = True
                dashboard_info["reason"] = "Viewer - marked as visible_to_viewer"
            elif user_role.lower() == 'departmental' and d.allowed_departments and user_department:
                if user_department in d.allowed_departments:
                    dashboard_info["visible_to_current_user"] = True
                    dashboard_info["reason"] = f"Departmental - {user_department} in allowed list"
            
            debug_info["dashboards"].append(dashboard_info)
        
        return JSONResponse(debug_info)
        
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)
    finally:
        db.close()

# API: Get dashboard history for current user (with RBAC filtering)
@app.get("/api/get-dashboards")
async def get_dashboards(session: dict = Depends(require_auth)):
    """Get dashboards visible to current user based on RBAC rules"""
    db = SessionLocal()
    try:
        user_role = session['role']
        user_department = session.get('department')
        user_id = session['user_id']
        
        # Build query based on role
        if user_role.lower() == 'admin':
            # Admin: See only their own dashboards
            dashboards = db.query(Dashboard).filter(
                Dashboard.user_id == user_id,
                Dashboard.is_active == True
            ).order_by(Dashboard.updated_at.desc()).all()
            
        elif user_role.lower() == 'analyst':
            # Analyst: See only their own dashboards
            dashboards = db.query(Dashboard).filter(
                Dashboard.user_id == user_id,
                Dashboard.is_active == True
            ).order_by(Dashboard.updated_at.desc()).all()
            
        elif user_role.lower() == 'departmental':
            # Departmental: See their own dashboards + dashboards from other departmental users in same dept
            # Only see departmental dashboards that are either:
            # 1. Created by them
            # 2. Created by another departmental user in the same department
            # 3. Explicitly shared with their department via allowed_departments
            dashboards = db.query(Dashboard).filter(
                Dashboard.is_active == True
            ).filter(
                (Dashboard.user_id == user_id) |  # Their own dashboards
                ((Dashboard.created_by_role == 'Departmental') & 
                 (Dashboard.allowed_departments.like(f'%{user_department}%')))  # Dept dashboards shared with their dept
            ).order_by(Dashboard.updated_at.desc()).all()
            
        elif user_role.lower() == 'viewer':
            # Viewer: See only dashboards marked as visible_to_viewer
            print(f"🔍 Fetching dashboards for VIEWER role...")
            dashboards = db.query(Dashboard).filter(
                Dashboard.is_active == True,
                Dashboard.visible_to_viewer == True
            ).order_by(Dashboard.updated_at.desc()).all()
            print(f"   Found {len(dashboards)} dashboards visible to viewers")
            for d in dashboards:
                print(f"   - ID: {d.id}, Title: {d.title}, visible_to_viewer: {d.visible_to_viewer}")
            
        else:
            # Unknown role: no dashboards
            dashboards = []
        
        # Build response
        dashboard_list = []
        for d in dashboards:
            # Get creator info for display
            creator = db.query(User).filter(User.id == d.user_id).first()
            creator_name = creator.full_name if creator else "Unknown"
            
            dashboard_list.append({
                "id": d.id,
                "title": d.title,
                "description": d.description,
                "chart_count": d.chart_count,
                "file_path": d.file_path,
                "created_by": creator_name,
                "created_by_role": d.created_by_role,
                "is_owner": d.user_id == user_id,
                "visible_to_viewer": d.visible_to_viewer,
                "allowed_departments": d.allowed_departments.split(',') if d.allowed_departments else [],
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None
            })
        
        return JSONResponse({
            "success": True,
            "dashboards": dashboard_list,
            "count": len(dashboard_list),
            "user_role": user_role
        })
        
    except Exception as e:
        print(f"Error fetching dashboards: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error fetching dashboards: {str(e)}"
        }, status_code=500)
    finally:
        db.close()

# API: Load dashboard by ID
@app.get("/api/load-dashboard/{dashboard_id}")
async def load_dashboard(dashboard_id: int, session: dict = Depends(require_auth)):
    """Load a specific dashboard configuration with RBAC checks"""
    db = SessionLocal()
    try:
        user_role = session['role']
        user_department = session.get('department')
        user_id = session['user_id']
        
        # Get dashboard by ID
        dashboard = db.query(Dashboard).filter(
            Dashboard.id == dashboard_id,
            Dashboard.is_active == True
        ).first()
        
        if not dashboard:
            return JSONResponse({
                "success": False,
                "message": "Dashboard not found"
            }, status_code=404)
        
        # RBAC: Check if user has access to load this dashboard
        has_access = False
        
        # Owner always has access
        if dashboard.user_id == user_id:
            has_access = True
        # Admin can access all dashboards
        elif user_role.lower() == 'admin':
            has_access = True
        # Viewer can access dashboards marked as visible_to_viewer
        elif user_role.lower() == 'viewer' and dashboard.visible_to_viewer:
            has_access = True
        # Departmental can access:
        # 1. Dashboards created by departmental users and shared with their department
        elif user_role.lower() == 'departmental' and user_department:
            if dashboard.created_by_role == 'Departmental' and dashboard.allowed_departments:
                if user_department in dashboard.allowed_departments:
                    has_access = True
        
        if not has_access:
            return JSONResponse({
                "success": False,
                "message": "Access denied: You don't have permission to load this dashboard"
            }, status_code=403)
        
        # Parse charts config
        import json
        charts_config = json.loads(dashboard.charts_config)
        
        return JSONResponse({
            "success": True,
            "dashboard": {
                "id": dashboard.id,
                "title": dashboard.title,
                "description": dashboard.description,
                "charts_config": charts_config,
                "chart_count": dashboard.chart_count,
                "file_path": dashboard.file_path,
                "created_at": dashboard.created_at.isoformat() if dashboard.created_at else None,
                "updated_at": dashboard.updated_at.isoformat() if dashboard.updated_at else None
            }
        })
        
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error loading dashboard: {str(e)}"
        }, status_code=500)
    finally:
        db.close()

# API: Update dashboard
@app.put("/api/update-dashboard/{dashboard_id}")
async def update_dashboard(
    dashboard_id: int,
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Update an existing dashboard"""
    db = SessionLocal()
    try:
        # Get dashboard by ID
        dashboard = db.query(Dashboard).filter(
            Dashboard.id == dashboard_id,
            Dashboard.user_id == session['user_id'],
            Dashboard.is_active == True
        ).first()
        
        if not dashboard:
            return JSONResponse({
                "success": False,
                "message": "Dashboard not found or access denied"
            }, status_code=404)
        
        # Update fields
        title = request_data.get('title', '').strip()
        description = request_data.get('description', '').strip()
        charts_config = request_data.get('charts_config', [])
        file_path = request_data.get('file_path')
        
        if title:
            dashboard.title = title
        if description is not None:
            dashboard.description = description
        if charts_config:
            import json
            dashboard.charts_config = json.dumps(charts_config)
            dashboard.chart_count = len(charts_config)
        if file_path:
            dashboard.file_path = file_path
        
        db.commit()
        db.refresh(dashboard)
        
        return JSONResponse({
            "success": True,
            "message": f"Dashboard '{dashboard.title}' updated successfully!"
        })
        
    except Exception as e:
        print(f"Error updating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error updating dashboard: {str(e)}"
        }, status_code=500)
    finally:
        db.close()

# API: Delete dashboard
@app.delete("/api/delete-dashboard/{dashboard_id}")
async def delete_dashboard_api(dashboard_id: int, session: dict = Depends(require_auth)):
    """Delete a dashboard (soft delete)"""
    db = SessionLocal()
    try:
        # Get dashboard by ID
        dashboard = db.query(Dashboard).filter(
            Dashboard.id == dashboard_id,
            Dashboard.user_id == session['user_id'],
            Dashboard.is_active == True
        ).first()
        
        if not dashboard:
            return JSONResponse({
                "success": False,
                "message": "Dashboard not found or access denied"
            }, status_code=404)
        
        # Soft delete
        dashboard.is_active = False
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": f"Dashboard '{dashboard.title}' deleted successfully!"
        })
        
    except Exception as e:
        print(f"Error deleting dashboard: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": f"Error deleting dashboard: {str(e)}"
        }, status_code=500)
    finally:
        db.close()

# API: Export dashboard data as CSV/Excel/JSON
@app.post("/api/export-dashboard-data")
async def export_dashboard_data(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Export dashboard data in various formats (CSV, Excel, JSON) for Tableau or other BI tools"""
    db = SessionLocal()
    try:
        dashboard_id = request_data.get('dashboard_id')
        export_format = request_data.get('format', 'excel').lower()  # excel, csv, json
        
        if not dashboard_id:
            return JSONResponse({
                "success": False,
                "error": "Dashboard ID is required"
            }, status_code=400)
        
        # Get dashboard by ID
        dashboard = db.query(Dashboard).filter(
            Dashboard.id == dashboard_id,
            Dashboard.user_id == session['user_id'],
            Dashboard.is_active == True
        ).first()
        
        if not dashboard:
            return JSONResponse({
                "success": False,
                "error": "Dashboard not found or access denied"
            }, status_code=404)
        
        # Parse charts config to extract data
        charts_config = json.loads(dashboard.charts_config)
        
        # Create exports directory if it doesn't exist
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"dashboard_{dashboard_id}_data_{timestamp}"
        
        # Get dashboard system to access data
        ds = get_dashboard_system()
        if not ds or not ds.nlu_pipeline or not ds.nlu_pipeline.chart_generator:
            return JSONResponse({
                "success": False,
                "error": "Dashboard system not available"
            }, status_code=500)
        
        data_connector = ds.nlu_pipeline.chart_generator.data_connector
        
        if export_format == 'csv':
            # Export as CSV files (one per chart)
            csv_files = []
            for idx, chart_config in enumerate(charts_config):
                query = chart_config.get('query', '')
                
                try:
                    # Re-process query to get fresh data
                    result = ds.nlu_pipeline.process_query(query)
                    if result and result[0]:
                        fig = result[0]
                        
                        # Extract data from figure
                        combined_df = pd.DataFrame()
                        for trace in fig.data:
                            trace_data = {}
                            if hasattr(trace, 'x') and trace.x is not None:
                                trace_data['x'] = list(trace.x)
                            if hasattr(trace, 'y') and trace.y is not None:
                                trace_data['y'] = list(trace.y)
                            if hasattr(trace, 'name'):
                                trace_data['series'] = trace.name
                            
                            if trace_data:
                                temp_df = pd.DataFrame(trace_data)
                                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
                        
                        if not combined_df.empty:
                            csv_filename = f"{base_filename}_chart{idx+1}.csv"
                            csv_path = export_dir / csv_filename
                            combined_df.to_csv(csv_path, index=False)
                            csv_files.append(str(csv_path))
                except Exception as e:
                    print(f"Error exporting chart {idx+1}: {e}")
                    continue
            
            if len(csv_files) == 0:
                return JSONResponse({
                    "success": False,
                    "error": "No data available to export"
                }, status_code=400)
            elif len(csv_files) == 1:
                # Single file - return it directly
                return FileResponse(
                    csv_files[0],
                    media_type="text/csv",
                    filename=Path(csv_files[0]).name
                )
            else:
                # Multiple files - create zip
                zip_filename = f"{base_filename}.zip"
                zip_path = export_dir / zip_filename
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for csv_file in csv_files:
                        zipf.write(csv_file, Path(csv_file).name)
                
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename=zip_filename
                )
        
        elif export_format == 'excel':
            # Export as Excel file with multiple sheets
            excel_filename = f"{base_filename}.xlsx"
            excel_path = export_dir / excel_filename
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Add dashboard info sheet
                info_df = pd.DataFrame({
                    'Property': ['Dashboard Title', 'Description', 'Created By', 'Total Charts', 'Export Date'],
                    'Value': [
                        dashboard.title,
                        dashboard.description or 'N/A',
                        session['full_name'],
                        dashboard.chart_count,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                })
                info_df.to_excel(writer, sheet_name='Dashboard Info', index=False)
                
                # Add each chart's data as a separate sheet
                for idx, chart_config in enumerate(charts_config):
                    query = chart_config.get('query', '')
                    
                    try:
                        # Re-process query to get fresh data
                        result = ds.nlu_pipeline.process_query(query)
                        if result and result[0]:
                            fig = result[0]
                            
                            # Extract data from figure
                            combined_df = pd.DataFrame()
                            for trace in fig.data:
                                trace_data = {}
                                if hasattr(trace, 'x') and trace.x is not None:
                                    trace_data['x'] = list(trace.x)
                                if hasattr(trace, 'y') and trace.y is not None:
                                    trace_data['y'] = list(trace.y)
                                if hasattr(trace, 'name'):
                                    trace_data['series'] = trace.name
                                
                                if trace_data:
                                    temp_df = pd.DataFrame(trace_data)
                                    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
                            
                            if not combined_df.empty:
                                sheet_name = f"Chart_{idx+1}"[:31]  # Excel sheet name limit
                                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    except Exception as e:
                        print(f"Error exporting chart {idx+1} to Excel: {e}")
                        continue
            
            return FileResponse(
                excel_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=excel_filename
            )
        
        elif export_format == 'json':
            # Export as JSON file
            json_data = {
                "dashboard_info": {
                    "id": dashboard.id,
                    "title": dashboard.title,
                    "description": dashboard.description,
                    "created_by": session['full_name'],
                    "chart_count": dashboard.chart_count,
                    "export_date": datetime.now().isoformat()
                },
                "charts": []
            }
            
            for idx, chart_config in enumerate(charts_config):
                query = chart_config.get('query', '')
                
                try:
                    # Re-process query to get fresh data
                    result = ds.nlu_pipeline.process_query(query)
                    if result and result[0]:
                        fig = result[0]
                        
                        chart_data = {
                            "chart_number": idx + 1,
                            "query": query,
                            "title": fig.layout.title.text if fig.layout.title else f"Chart {idx+1}",
                            "data": []
                        }
                        
                        # Extract data from figure
                        for trace in fig.data:
                            trace_data = {
                                "name": trace.name if hasattr(trace, 'name') else None,
                                "type": trace.type if hasattr(trace, 'type') else None,
                                "x": list(trace.x) if hasattr(trace, 'x') and trace.x is not None else [],
                                "y": list(trace.y) if hasattr(trace, 'y') and trace.y is not None else []
                            }
                            chart_data["data"].append(trace_data)
                        
                        json_data["charts"].append(chart_data)
                except Exception as e:
                    print(f"Error exporting chart {idx+1} to JSON: {e}")
                    continue
            
            json_filename = f"{base_filename}.json"
            json_path = export_dir / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            return FileResponse(
                json_path,
                media_type="application/json",
                filename=json_filename
            )
        
        else:
            return JSONResponse({
                "success": False,
                "error": f"Invalid export format: {export_format}. Use 'csv', 'excel', or 'json'."
            }, status_code=400)
        
    except Exception as e:
        print(f"Error exporting dashboard data: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error exporting data: {str(e)}"
        }, status_code=500)
    finally:
        db.close()

# API: Export raw dataset tables
@app.post("/api/export-dataset-tables")
async def export_dataset_tables(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Export raw dataset tables for Tableau or other BI tools"""
    try:
        table_names = request_data.get('tables', [])  # List of table names to export
        export_format = request_data.get('format', 'excel').lower()
        
        # Get dashboard system
        ds = get_dashboard_system()
        if not ds or not ds.nlu_pipeline or not ds.nlu_pipeline.chart_generator:
            return JSONResponse({
                "success": False,
                "error": "Dashboard system not available"
            }, status_code=500)
        
        data_connector = ds.nlu_pipeline.chart_generator.data_connector
        
        # Create exports directory
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"dataset_tables_{timestamp}"
        
        # If no specific tables requested, export all
        if not table_names:
            table_names = list(data_connector.cached_data.keys())
        
        if export_format == 'excel':
            # Export all tables to one Excel file with multiple sheets
            excel_filename = f"{base_filename}.xlsx"
            excel_path = export_dir / excel_filename
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for table_name in table_names:
                    if table_name in data_connector.cached_data:
                        df = data_connector.cached_data[table_name]
                        sheet_name = table_name[:31]  # Excel limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            return FileResponse(
                excel_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=excel_filename
            )
        
        elif export_format == 'csv':
            # Export as CSV files in a zip
            csv_files = []
            for table_name in table_names:
                if table_name in data_connector.cached_data:
                    df = data_connector.cached_data[table_name]
                    csv_filename = f"{table_name}.csv"
                    csv_path = export_dir / csv_filename
                    df.to_csv(csv_path, index=False)
                    csv_files.append(str(csv_path))
            
            if len(csv_files) == 0:
                return JSONResponse({
                    "success": False,
                    "error": "No tables found to export"
                }, status_code=400)
            elif len(csv_files) == 1:
                return FileResponse(
                    csv_files[0],
                    media_type="text/csv",
                    filename=Path(csv_files[0]).name
                )
            else:
                zip_filename = f"{base_filename}.zip"
                zip_path = export_dir / zip_filename
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for csv_file in csv_files:
                        zipf.write(csv_file, Path(csv_file).name)
                
                return FileResponse(
                    zip_path,
                    media_type="application/zip",
                    filename=zip_filename
                )
        
        else:
            return JSONResponse({
                "success": False,
                "error": f"Invalid export format: {export_format}. Use 'csv' or 'excel'."
            }, status_code=400)
        
    except Exception as e:
        print(f"Error exporting dataset tables: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error exporting dataset: {str(e)}"
        }, status_code=500)

# API: Export dashboard as standalone HTML
@app.get("/api/export-dashboard-html/{dashboard_id}")
async def export_dashboard_html(dashboard_id: int, session: dict = Depends(require_auth)):
    """Export dashboard as standalone HTML file"""
    db = SessionLocal()
    try:
        dashboard = db.query(Dashboard).filter(
            Dashboard.id == dashboard_id,
            Dashboard.user_id == session['user_id'],
            Dashboard.is_active == True
        ).first()
        
        if not dashboard:
            return JSONResponse({
                "success": False,
                "error": "Dashboard not found or access denied"
            }, status_code=404)
        
        # Check if dashboard file exists
        dashboard_file = Path(dashboard.file_path)
        if not dashboard_file.exists():
            return JSONResponse({
                "success": False,
                "error": "Dashboard file not found"
            }, status_code=404)
        
        # Create exports directory
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        
        # Copy to exports with a clean filename
        export_filename = f"{dashboard.title.replace(' ', '_')}_{dashboard_id}.html"
        export_path = export_dir / export_filename
        
        import shutil
        shutil.copy(dashboard_file, export_path)
        
        return FileResponse(
            export_path,
            media_type="text/html",
            filename=export_filename
        )
        
    except Exception as e:
        print(f"Error exporting dashboard HTML: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error exporting dashboard: {str(e)}"
        }, status_code=500)
    finally:
        db.close()

# API: Generate single chart (for chart creator)
@app.post("/api/generate-single-chart")
async def generate_single_chart(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Generate a single chart from query without adding to dashboard"""
    # Capture print statements to return as processing logs
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    log_buffer = io.StringIO()
    
    try:
        query = request_data.get('query', '')
        if query:
            query = query.strip()
        chart_type = request_data.get('chart_type', 'auto')
        title = request_data.get('title', '')
        if title:
            title = title.strip()
        
        if not query:
            return JSONResponse({
                "success": False,
                "error": "Query cannot be empty"
            }, status_code=400)
        
        # Get dashboard system to use its NLU capabilities
        ds = get_dashboard_system()
        if not ds:
            return JSONResponse({
                "success": False,
                "error": "Dashboard system not available. Make sure the NLU model is trained."
            }, status_code=500)
        
        # Check if NLU pipeline is available
        if not ds.nlu_pipeline:
            return JSONResponse({
                "success": False,
                "error": "NLU pipeline not available. Please ensure the system is properly initialized."
            }, status_code=500)
        
        # If user selected a specific chart type (not auto), append it to the query
        modified_query = query
        if chart_type and chart_type != 'auto':
            # Append chart type to query so it can be detected
            modified_query = f"{query} as {chart_type} chart"
            log_msg = f"📊 Modified query with chart type: {modified_query}"
            print(log_msg)
            log_buffer.write(log_msg + "\n")
        
        print(f"\n{'='*60}")
        print(f"🎨 CHART CREATOR - Processing Query: {modified_query}")
        print(f"{'='*60}\n")
        
        # Capture stdout during processing
        with redirect_stdout(log_buffer):
            # Process query using NLU pipeline
            result = ds.nlu_pipeline.process_query(modified_query)
        
        # Print captured logs to terminal for debugging
        captured_output = log_buffer.getvalue()
        if captured_output:
            print("📋 Captured Processing Logs:")
            print(captured_output)
            print("="*60)
        
        if not result or not result[0]:
            return JSONResponse({
                "success": False,
                "error": "Could not understand the query. Please rephrase.",
                "processing_logs": log_buffer.getvalue()
            }, status_code=400)
        
        # Extract the figure and entities from the result tuple (fig, entities, tokens, labels)
        fig = result[0]
        entities = result[1] if len(result) > 1 else {}
        
        # If user specified a custom title, update it
        if title:
            fig.update_layout(title=title)
        
        if fig:
            # Helper function to convert numpy/pandas types to native Python types
            def convert_to_json_serializable(obj):
                """Convert numpy/pandas types to JSON serializable Python types"""
                import numpy as np
                import pandas as pd
                
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                elif isinstance(obj, (pd.Series, pd.Index)):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            # Convert Plotly figure to JSON-compatible format
            import plotly.io as pio
            chart_json = pio.to_json(fig)
            
            # Parse the JSON string back to dict for the response
            import json
            chart_data = json.loads(chart_json)
            
            # Get processing logs
            processing_logs = log_buffer.getvalue()
            
            # Extract data preview from the figure
            data_preview = None
            try:
                # Get the data from the Plotly figure
                if fig.data and len(fig.data) > 0:
                    trace = fig.data[0]
                    
                    # Extract columns and data based on chart type
                    columns = []
                    rows = []
                    
                    # Handle different chart types
                    if hasattr(trace, 'x') and hasattr(trace, 'y'):
                        # Bar, Line, Scatter charts
                        x_data = [convert_to_json_serializable(x) for x in trace.x] if trace.x is not None else []
                        y_data = [convert_to_json_serializable(y) for y in trace.y] if trace.y is not None else []
                        
                        # Get axis titles or use defaults
                        x_label = fig.layout.xaxis.title.text if fig.layout.xaxis.title else 'X'
                        y_label = fig.layout.yaxis.title.text if fig.layout.yaxis.title else 'Y'
                        
                        columns = [str(x_label), str(y_label)]
                        
                        # Create rows with converted values
                        for i in range(min(len(x_data), len(y_data))):
                            rows.append({
                                str(x_label): x_data[i],
                                str(y_label): y_data[i]
                            })
                    
                    elif hasattr(trace, 'labels') and hasattr(trace, 'values'):
                        # Pie charts
                        labels = [convert_to_json_serializable(lbl) for lbl in trace.labels] if trace.labels is not None else []
                        values = [convert_to_json_serializable(val) for val in trace.values] if trace.values is not None else []
                        
                        columns = ['Category', 'Value']
                        
                        for i in range(min(len(labels), len(values))):
                            rows.append({
                                'Category': labels[i],
                                'Value': values[i]
                            })
                    
                    if columns and rows:
                        data_preview = {
                            'columns': columns,
                            'data': rows
                        }
            except Exception as e:
                print(f"⚠️ Could not extract data preview: {e}")
                import traceback
                traceback.print_exc()
            
            return JSONResponse({
                "success": True,
                "chart_data": chart_data,
                "data_preview": data_preview,
                "query": query,
                "processing_logs": processing_logs,
                "extracted_entities": entities
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Failed to generate chart",
                "processing_logs": log_buffer.getvalue()
            }, status_code=500)
            
    except Exception as e:
        error_msg = f"Error generating single chart: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "success": False,
            "error": f"Error: {str(e)}",
            "processing_logs": log_buffer.getvalue() + "\n" + traceback.format_exc()
        }, status_code=500)

# ========== DASHBOARD EXPLAINABILITY ENDPOINTS ==========

# API: Upload company profile for explainability
@app.post("/api/upload-company-profile")
async def upload_company_profile(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Upload company profile document for RAG-based dashboard explanations"""
    try:
        if not get_dashboard_explainer:
            return JSONResponse({
                "success": False,
                "error": "Dashboard explainer not available"
            }, status_code=500)
        
        file_url = request_data.get('file_url')
        file_content = request_data.get('file_content')
        
        if not file_url and not file_content:
            return JSONResponse({
                "success": False,
                "error": "Please provide file_url or file_content"
            }, status_code=400)
        
        explainer = get_dashboard_explainer()
        
        # If file content is provided, save it temporarily
        if file_content:
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            temp_file = temp_dir / f"company_profile_{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            success = explainer.load_company_profile(str(temp_file))
        else:
            success = explainer.load_company_profile(file_url)
        
        if success:
            # Initialize RAG system
            explainer.initialize_rag_system()
            
            return JSONResponse({
                "success": True,
                "message": "Company profile loaded successfully"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Failed to load company profile"
            }, status_code=500)
            
    except Exception as e:
        print(f"Error uploading company profile: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error: {str(e)}"
        }, status_code=500)

# API: Load dataset info for explainability
@app.post("/api/load-dataset-for-explanation")
async def load_dataset_for_explanation(session: dict = Depends(require_auth)):
    """Load current dataset information into explainer"""
    try:
        if not get_dashboard_explainer:
            return JSONResponse({
                "success": False,
                "error": "Dashboard explainer not available"
            }, status_code=500)
        
        # Get dataset info from dashboard system
        ds = get_dashboard_system()
        if not ds or not ds.nlu_pipeline or not ds.nlu_pipeline.chart_generator:
            return JSONResponse({
                "success": False,
                "error": "Dashboard system not initialized"
            }, status_code=500)
        
        data_connector = ds.nlu_pipeline.chart_generator.data_connector
        columns_info = data_connector.extract_all_columns_info()
        
        # Format dataset info
        dataset_info = {"tables": []}
        for table_name, info in columns_info.items():
            table_data = {
                "name": table_name,
                "row_count": len(data_connector.cached_data.get(table_name, [])),
                "columns": [
                    {"name": col, "type": str(dtype)}
                    for col, dtype in info.items()
                ]
            }
            dataset_info["tables"].append(table_data)
        
        # Load into explainer
        explainer = get_dashboard_explainer()
        success = explainer.load_dataset_info(dataset_info)
        
        if success:
            # Initialize RAG system
            explainer.initialize_rag_system()
            
            return JSONResponse({
                "success": True,
                "message": "Dataset information loaded successfully"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Failed to load dataset information"
            }, status_code=500)
            
    except Exception as e:
        print(f"Error loading dataset for explanation: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error: {str(e)}"
        }, status_code=500)

# API: Explain dashboard
@app.post("/api/explain-dashboard")
async def explain_dashboard_api(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Generate RAG-based explanations for a dashboard"""
    try:
        if not get_dashboard_explainer:
            return JSONResponse({
                "success": False,
                "error": "Dashboard explainer not available"
            }, status_code=500)
        
        dashboard_id = request_data.get('dashboard_id')
        charts_config = request_data.get('charts_config')
        force_chart_only = request_data.get('force_chart_only', False)
        
        if not charts_config and not dashboard_id:
            return JSONResponse({
                "success": False,
                "error": "Please provide either dashboard_id or charts_config"
            }, status_code=400)
        
        # If dashboard_id provided, load from database
        if dashboard_id and not charts_config:
            db = SessionLocal()
            try:
                dashboard = db.query(Dashboard).filter(
                    Dashboard.id == dashboard_id,
                    Dashboard.user_id == session['user_id']
                ).first()
                
                if not dashboard:
                    return JSONResponse({
                        "success": False,
                        "error": "Dashboard not found"
                    }, status_code=404)
                
                charts_config = json.loads(dashboard.charts_config)
            finally:
                db.close()
        
        # Prepare dashboard config
        dashboard_config = {
            "title": request_data.get('title', 'Analytics Dashboard'),
            "charts": charts_config
        }
        
        # Generate explanations
        explainer = get_dashboard_explainer()
        result = explainer.explain_dashboard(dashboard_config, force_chart_only=force_chart_only)
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"Error explaining dashboard: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error: {str(e)}"
        }, status_code=500)

# API: Explain single chart
@app.post("/api/explain-chart")
async def explain_chart_api(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Generate explanation for a single chart"""
    try:
        if not get_dashboard_explainer:
            return JSONResponse({
                "success": False,
                "error": "Dashboard explainer not available"
            }, status_code=500)
        
        chart_config = request_data.get('chart')
        
        if not chart_config:
            return JSONResponse({
                "success": False,
                "error": "Please provide chart configuration"
            }, status_code=400)
        
        explainer = get_dashboard_explainer()
        result = explainer.explain_single_chart(chart_config)
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"Error explaining chart: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error: {str(e)}"
        }, status_code=500)

# API: Get comparative insights
@app.post("/api/get-comparative-insights")
async def get_comparative_insights_api(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Get comparative insights across multiple charts"""
    try:
        if not get_dashboard_explainer:
            return JSONResponse({
                "success": False,
                "error": "Dashboard explainer not available"
            }, status_code=500)
        
        charts = request_data.get('charts', [])
        
        if not charts:
            return JSONResponse({
                "success": False,
                "error": "Please provide charts configuration"
            }, status_code=400)
        
        explainer = get_dashboard_explainer()
        result = explainer.get_comparative_insights(charts)
        
        return JSONResponse(result)
        
    except Exception as e:
        print(f"Error getting comparative insights: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error: {str(e)}"
        }, status_code=500)

# API: Check explainer status
@app.get("/api/explainer-status")
async def explainer_status_api(session: dict = Depends(require_auth)):
    """Check if explainer is initialized and ready"""
    try:
        if not get_dashboard_explainer:
            return JSONResponse({
                "initialized": False,
                "ready": False,
                "message": "Dashboard explainer not available"
            })
        
        explainer = get_dashboard_explainer()
        
        has_company_profile = explainer.company_docs is not None
        has_dataset = explainer.dataset_docs is not None
        has_vectorstore = explainer.vectorstore is not None
        
        return JSONResponse({
            "initialized": True,
            "ready": has_vectorstore,
            "has_company_profile": has_company_profile,
            "has_dataset": has_dataset,
            "message": "Explainer ready" if has_vectorstore else "Please upload company profile and load dataset"
        })
        
    except Exception as e:
        return JSONResponse({
            "initialized": False,
            "ready": False,
            "error": str(e)
        })

# ========== RAG FILE UPLOAD ENDPOINTS ==========

# API: Upload file for RAG processing
@app.post("/api/upload-file-for-rag")
async def upload_file_for_rag(
    file: UploadFile = File(...),
    session: dict = Depends(require_auth)
):
    """Upload file (PDF, DOCX, TXT, etc.) for RAG-based explainability"""
    try:
        # Create temp_uploads directory if it doesn't exist
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix
        safe_filename = f"rag_upload_{session['user_id']}_{timestamp}{file_extension}"
        file_path = temp_dir / safe_filename
        
        # Write file content
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Get file size
        file_size = len(content)
        
        return JSONResponse({
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully",
            "file_path": str(file_path),
            "file_size": file_size,
            "filename": file.filename
        })
        
    except Exception as e:
        print(f"Error uploading file for RAG: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error uploading file: {str(e)}"
        }, status_code=500)

# API: Process uploaded file with RAG system
@app.post("/api/process-rag-document")
async def process_rag_document(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Process uploaded document with RAG system for explainability"""
    try:
        from rag.rag import process_document_and_questions
        
        file_path = request_data.get('file_path')
        questions = request_data.get('questions', [])
        
        if not file_path:
            return JSONResponse({
                "success": False,
                "error": "file_path is required"
            }, status_code=400)
        
        if not questions or len(questions) == 0:
            # Default questions for dashboard explainability
            questions = [
                "What are the key insights from this dataset?",
                "What patterns or trends are visible in the data?",
                "What are the main data quality issues or anomalies?",
                "What business recommendations can be derived from this data?"
            ]
        
        # Process document with RAG
        answers = process_document_and_questions(file_path, questions)
        
        # Format response
        qa_pairs = []
        for q, a in zip(questions, answers):
            qa_pairs.append({
                "question": q,
                "answer": a
            })
        
        return JSONResponse({
            "success": True,
            "message": "Document processed successfully",
            "qa_pairs": qa_pairs,
            "document_path": file_path
        })
        
    except Exception as e:
        print(f"Error processing RAG document: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error processing document: {str(e)}"
        }, status_code=500)

# API: Query RAG system
@app.post("/api/query-rag")
async def query_rag_system(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Query the RAG system with custom questions about uploaded documents"""
    try:
        from rag.rag import process_document_and_questions
        
        file_path = request_data.get('file_path')
        question = request_data.get('question')
        
        if not file_path or not question:
            return JSONResponse({
                "success": False,
                "error": "file_path and question are required"
            }, status_code=400)
        
        # Process single question
        answers = process_document_and_questions(file_path, [question])
        
        return JSONResponse({
            "success": True,
            "question": question,
            "answer": answers[0] if answers else "No answer generated"
        })
        
    except Exception as e:
        print(f"Error querying RAG system: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error: {str(e)}"
        }, status_code=500)

# ========== END RAG FILE UPLOAD ENDPOINTS ==========

# ========== END EXPLAINABILITY ENDPOINTS ==========

# API: Save chart to session
@app.post("/api/save-chart")
async def save_chart(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth)
):
    """Save a chart to the user's session dashboard"""
    try:
        query = request_data.get('query', '')
        if query:
            query = query.strip()
        chart_data = request_data.get('chart_data')
        
        if not query or not chart_data:
            return JSONResponse({
                "success": False,
                "error": "Query and chart data are required"
            }, status_code=400)
        
        # Get dashboard system
        ds = get_dashboard_system()
        if not ds:
            return JSONResponse({
                "success": False,
                "error": "Dashboard system not available"
            }, status_code=500)
        
        # Add chart using the query
        success = ds.add_chart_from_query(query)
        
        if success:
            chart_count = ds.dashboard.get_chart_count()
            return JSONResponse({
                "success": True,
                "message": f"Chart saved! Total charts: {chart_count}",
                "chart_count": chart_count
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Failed to save chart"
            }, status_code=400)
            
    except Exception as e:
        print(f"Error saving chart: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error: {str(e)}"
        }, status_code=500)

def get_password_hash(password: str) -> str:
    # Truncate password to 72 bytes for bcrypt compatibility
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Truncate password to 72 bytes for bcrypt compatibility
    password_bytes = plain_password.encode('utf-8')[:72]
    return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))

@app.post("/register")
async def register(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    department: Optional[str] = Form(None)
):
    db = SessionLocal()
    try:
        user_data = UserCreate(
            full_name=full_name,
            email=email,
            username=username,
            password=password,
            role=role,
            department=department
        )
        
        existing = db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        if existing:
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Registration Failed</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                        margin: 5px;
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1>Registration Failed!</h1>
                    <p>Username or email already exists. Please try with different credentials.</p>
                    <a href="/register.html" class="btn">Try Again</a>
                    <a href="/page.html" class="btn" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Login Instead</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=400)
        
        db_user = User(
            full_name=user_data.full_name,
            email=user_data.email,
            username=user_data.username,
            hashed_password=get_password_hash(user_data.password),
            role=user_data.role,
            department=user_data.department
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        success_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Registration Successful</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .success-container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 600px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(-50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .success-icon {{
                    font-size: 100px;
                    margin-bottom: 20px;
                    animation: bounce 1s ease-in-out;
                }}
                @keyframes bounce {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.2); }}
                }}
                h1 {{
                    color: #27ae60;
                    margin-bottom: 15px;
                    font-size: 32px;
                }}
                .subtitle {{
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 18px;
                }}
                .user-info {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    text-align: left;
                }}
                .user-info h3 {{
                    color: #333;
                    margin-bottom: 15px;
                    font-size: 20px;
                    text-align: center;
                }}
                .info-row {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #ddd;
                }}
                .info-row:last-child {{
                    border-bottom: none;
                }}
                .info-label {{
                    font-weight: 600;
                    color: #555;
                }}
                .info-value {{
                    color: #333;
                }}
                .success-message {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    font-size: 16px;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    margin: 5px;
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="success-container">
                <div class="success-icon">🎉</div>
                <h1>Registration Successful!</h1>
                <p class="subtitle">Your account has been created successfully!</p>
                
                <div class="success-message">
                    ✅ Data has been saved to MySQL database
                </div>
                
                <div class="user-info">
                    <h3>📋 Your Account Details</h3>
                    <div class="info-row">
                        <span class="info-label">👤 Full Name:</span>
                        <span class="info-value">{db_user.full_name}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">🆔 Username:</span>
                        <span class="info-value">{db_user.username}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">📧 Email:</span>
                        <span class="info-value">{db_user.email}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">🎭 Role:</span>
                        <span class="info-value">{db_user.role}</span>
                    </div>
                    {f'<div class="info-row"><span class="info-label">🏢 Department:</span><span class="info-value">{db_user.department}</span></div>' if db_user.department else ''}
                </div>
                
                <a href="/page.html" class="btn">Login Now</a>
                <a href="/register.html" class="btn" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Register Another</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=success_html, status_code=200)
        
    except ValueError as e:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Validation Error</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .error-container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 500px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(-50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .error-icon {{
                    font-size: 80px;
                    margin-bottom: 20px;
                    animation: shake 0.5s ease-in-out;
                }}
                @keyframes shake {{
                    0%, 100% {{ transform: translateX(0); }}
                    25% {{ transform: translateX(-10px); }}
                    75% {{ transform: translateX(10px); }}
                }}
                h1 {{
                    color: #e74c3c;
                    margin-bottom: 15px;
                    font-size: 28px;
                }}
                p {{
                    color: #555;
                    margin-bottom: 30px;
                    font-size: 16px;
                    line-height: 1.6;
                }}
                .error-detail {{
                    background: #fff3cd;
                    border: 1px solid #ffc107;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    color: #856404;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <h1>Validation Error!</h1>
                <p>Please check the following error and try again:</p>
                <div class="error-detail">
                    {str(e)}
                </div>
                <a href="/register.html" class="btn">Try Again</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=422)
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Registration Error</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .error-container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 500px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(-50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .error-icon {{
                    font-size: 80px;
                    margin-bottom: 20px;
                }}
                h1 {{
                    color: #e74c3c;
                    margin-bottom: 15px;
                    font-size: 28px;
                }}
                p {{
                    color: #555;
                    margin-bottom: 30px;
                    font-size: 16px;
                    line-height: 1.6;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <div class="error-icon">❌</div>
                <h1>Registration Failed!</h1>
                <p>An unexpected error occurred. Please try again.</p>
                <a href="/register.html" class="btn">Try Again</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)
    finally:
        db.close()

@app.post("/login")
async def login(
    response: Response,
    request: Request,
    loginId: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    department: Optional[str] = Form(None)
):
    db = SessionLocal()
    try:
        # Find user by username or email
        db_user = db.query(User).filter(
            (User.username == loginId.lower()) | 
            (User.email == loginId.lower())
        ).first()
        
        if not db_user:
            # Return HTML error page
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Login Failed</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1>Login Failed!</h1>
                    <p>Details not matched. Please check your credentials and try again.</p>
                    <a href="/page.html" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=401)
        
        if not verify_password(password, db_user.hashed_password):
            # Return HTML error page
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Login Failed</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1>Login Failed!</h1>
                    <p>Details not matched. Please check your credentials and try again.</p>
                    <a href="/page.html" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=401)
        
        if not db_user.is_active:
            # Return HTML error page
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Account Inactive</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">🚫</div>
                    <h1>Account Inactive</h1>
                    <p>Your account is currently inactive. Please contact support.</p>
                    <a href="/page.html" class="btn">Back to Login</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=401)
        
        # Verify role matches
        if db_user.role != role:
            # Return HTML error page
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Role Mismatch</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1>Login Failed!</h1>
                    <p>Role mismatch. Details not matched. Please try again.</p>
                    <a href="/page.html" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=401)
        
        # For departmental users, verify department matches
        if role == "Departmental":
            if not department or db_user.department != department:
                # Return HTML error page
                error_html = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Department Mismatch</title>
                    <style>
                        * { margin: 0; padding: 0; box-sizing: border-box; }
                        body {
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            min-height: 100vh;
                            padding: 20px;
                        }
                        .error-container {
                            background: white;
                            border-radius: 20px;
                            padding: 40px;
                            max-width: 500px;
                            width: 100%;
                            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                            text-align: center;
                            animation: slideIn 0.5s ease-out;
                        }
                        @keyframes slideIn {
                            from { transform: translateY(-50px); opacity: 0; }
                            to { transform: translateY(0); opacity: 1; }
                        }
                        .error-icon {
                            font-size: 80px;
                            margin-bottom: 20px;
                            animation: shake 0.5s ease-in-out;
                        }
                        @keyframes shake {
                            0%, 100% { transform: translateX(0); }
                            25% { transform: translateX(-10px); }
                            75% { transform: translateX(10px); }
                        }
                        h1 {
                            color: #e74c3c;
                            margin-bottom: 15px;
                            font-size: 28px;
                        }
                        p {
                            color: #555;
                            margin-bottom: 30px;
                            font-size: 16px;
                            line-height: 1.6;
                        }
                        .btn {
                            display: inline-block;
                            padding: 12px 40px;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            text-decoration: none;
                            border-radius: 30px;
                            font-weight: 600;
                            transition: transform 0.3s, box-shadow 0.3s;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                        }
                        .btn:hover {
                            transform: translateY(-2px);
                            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                        }
                    </style>
                </head>
                <body>
                    <div class="error-container">
                        <div class="error-icon">❌</div>
                        <h1>Login Failed!</h1>
                        <p>Department mismatch. Details not matched. Please try again.</p>
                        <a href="/page.html" class="btn">Try Again</a>
                    </div>
                </body>
                </html>
                """
                return HTMLResponse(content=error_html, status_code=401)
        
        # If all validations pass, create session and redirect to home
        user_data = {
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email,
            "full_name": db_user.full_name,
            "role": db_user.role,
            "department": db_user.department
        }
        
        session_id = create_session(user_data)
        
        # Get accessible pages for this role
        accessible_pages = get_accessible_pages(db_user.role)
        pages_list = ", ".join(accessible_pages)
        
        success_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Login Successful</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .success-container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 600px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(-50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .success-icon {{
                    font-size: 100px;
                    margin-bottom: 20px;
                    animation: bounce 1s ease-in-out;
                }}
                @keyframes bounce {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.2); }}
                }}
                h1 {{
                    color: #27ae60;
                    margin-bottom: 15px;
                    font-size: 32px;
                }}
                .subtitle {{
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 18px;
                }}
                .user-info {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    text-align: left;
                }}
                .user-info h3 {{
                    color: #333;
                    margin-bottom: 15px;
                    font-size: 20px;
                }}
                .info-row {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #ddd;
                }}
                .info-row:last-child {{
                    border-bottom: none;
                }}
                .info-label {{
                    font-weight: 600;
                    color: #555;
                }}
                .info-value {{
                    color: #333;
                }}
                .access-info {{
                    background: #e8f5e9;
                    border-left: 4px solid #4caf50;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    text-align: left;
                }}
                .access-info h4 {{
                    color: #2e7d32;
                    margin-bottom: 10px;
                }}
                .access-info p {{
                    color: #555;
                    margin: 5px 0;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    margin: 5px;
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="success-container">
                <div class="success-icon">✅</div>
                <h1>Details Validated Great!</h1>
                <p class="subtitle">Welcome back, {db_user.full_name}!</p>
                
                <div class="user-info">
                    <h3>📋 Account Information</h3>
                    <div class="info-row">
                        <span class="info-label">👤 Username:</span>
                        <span class="info-value">{db_user.username}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">📧 Email:</span>
                        <span class="info-value">{db_user.email}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">🎭 Role:</span>
                        <span class="info-value">{db_user.role}</span>
                    </div>
                    {f'<div class="info-row"><span class="info-label">🏢 Department:</span><span class="info-value">{db_user.department}</span></div>' if db_user.department else ''}
                </div>
                
                <div class="access-info">
                    <h4>🔐 Your Access Permissions (RBAC)</h4>
                    <p><strong>You can access {len(accessible_pages)} page(s):</strong></p>
                    <p>✓ {pages_list}</p>
                    <p style="margin-top: 10px; font-size: 14px; color: #666;">
                        Role-Based Access Control ensures you only see pages relevant to your role, 
                        protecting sensitive data and maintaining privacy.
                    </p>
                </div>
                
                <a href="/home" class="btn">Go to Dashboard</a>
            </div>
        </body>
        </html>
        """
        
        response = HTMLResponse(content=success_html, status_code=200)
        response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=1800)  # 30 min
        return response
    finally:
        db.close()
