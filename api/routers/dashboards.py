"""Dashboard CRUD router for Gen-Dash API

This module handles all dashboard-related endpoints including:
- Dashboard generation and saving
- Dashboard listing with RBAC filtering
- Dashboard loading, updating, and deletion
- Dashboard viewing (HTML responses)
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, Body, Path as PathParam, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from sqlalchemy.orm import Session

# Import dependencies
from api.dependencies import require_auth, get_db, get_dashboard_system

# Import database models
from database.models import Dashboard, User

# Import auth helpers
from auth.auth import check_permission

# Create router
router = APIRouter()

# ============================================================================
# Dashboard Generation Endpoints
# ============================================================================

@router.post("/api/generate-dashboard")
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

# ============================================================================
# Dashboard Save Endpoint
# ============================================================================

@router.post("/api/save-dashboard")
async def save_dashboard(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """Save dashboard configuration to database"""
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

# ============================================================================
# Dashboard List Endpoint (with RBAC)
# ============================================================================

@router.get("/api/get-dashboards")
async def get_dashboards(
    session: dict = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """Get dashboards visible to current user based on RBAC rules"""
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

# ============================================================================
# Dashboard Load Endpoint
# ============================================================================

@router.get("/api/load-dashboard/{dashboard_id}")
async def load_dashboard(
    dashboard_id: int,
    session: dict = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """Load a specific dashboard configuration with RBAC checks"""
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

# ============================================================================
# Dashboard Update Endpoint
# ============================================================================

@router.put("/api/update-dashboard/{dashboard_id}")
async def update_dashboard(
    dashboard_id: int,
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """Update an existing dashboard"""
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

# ============================================================================
# Dashboard Delete Endpoint
# ============================================================================

@router.delete("/api/delete-dashboard/{dashboard_id}")
async def delete_dashboard_api(
    dashboard_id: int,
    session: dict = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """Delete a dashboard (soft delete)"""
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

# ============================================================================
# Dashboard Viewing Endpoints (HTML Responses)
# ============================================================================

@router.get("/view-dashboard", response_class=HTMLResponse)
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

@router.get("/view-saved-dashboard/{dashboard_id}", response_class=HTMLResponse)
async def view_saved_dashboard(
    dashboard_id: int,
    session: dict = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """View a specific saved dashboard by ID with RBAC checks"""
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
