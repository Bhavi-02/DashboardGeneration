"""Dashboard Explainability API router for Gen-Dash"""
from fastapi import APIRouter, Depends, Body, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from api.dependencies import require_auth, get_dashboard_system
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import json
import traceback

router = APIRouter()

# Database setup - reusing the same configuration as main.py
DATABASE_URL = "mysql+pymysql://root:dhruv123@localhost:3306/analytics_dashboard"
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=5,
    max_overflow=10,
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import dashboard explainer with graceful fallback
try:
    from dashboard_explainer import get_dashboard_explainer
    from database.models import Dashboard
    print("✅ Dashboard Explainer loaded successfully in explainability router")
except ImportError as e:
    print(f"⚠️ Warning: Could not import dashboard explainer in explainability router: {e}")
    get_dashboard_explainer = None
    Dashboard = None


@router.post("/upload-company-profile")
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


@router.post("/load-dataset-for-explanation")
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


@router.post("/explain-dashboard")
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


@router.post("/explain-chart")
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


@router.post("/get-comparative-insights")
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


@router.get("/explainer-status")
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
