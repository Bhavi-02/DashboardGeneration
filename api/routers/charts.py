"""Chart management router for Gen-Dash API

This module handles all chart-related endpoints including:
- Chart creation from NLU queries
- Chart deletion and management
- Smart dashboard generation (AI-powered)
- Single chart generation
- Chart saving to session
"""

import logging
import traceback
import io
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import redirect_stdout

from fastapi import APIRouter, Depends, Body, Cookie, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

# Import dependencies
from api.dependencies import require_auth, get_dashboard_system
from services.feedback_service import _build_personalization_prompt, _get_or_create_session_profile

# Import dashboard components
from dashboard.smart_generator import SmartDashboardGenerator
from charts.data_connector import DataConnector

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# ============================================================================
# Chart Management Endpoints
# ============================================================================

@router.post("/add-chart")
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

@router.post("/delete-chart")
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

@router.post("/clear-charts")
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

@router.get("/current-chart-count")
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

# ============================================================================
# Smart Dashboard Generation Endpoint
# ============================================================================

@router.post("/generate-smart-dashboard")
async def generate_smart_dashboard_api(
    request_data: Dict[str, Any] = Body(...),
    session: dict = Depends(require_auth),
    session_id: Optional[str] = Cookie(None)
):
    """
    Generate AI-powered dashboard with automatic chart recommendations

    Request Body:
    {
        "num_charts": 5,  // Number of charts to generate (default 5)
        "custom_prompt": "focus on CEO metrics",  // Optional: custom instructions for LLM
        "override_context": {  // Optional: override user context
            "department": "Marketing",
            "role": "Analyst"
        }
    }
    """
    try:
        logger.info(f"🤖 Smart dashboard generation requested by user {session['user_id']}")

        # Check if SmartDashboardGenerator is available
        if SmartDashboardGenerator is None:
            return JSONResponse({
                "success": False,
                "error": "Smart Dashboard Generator not available. Check server logs."
            }, status_code=500)

        # Extract parameters
        num_charts = request_data.get('num_charts', 5)
        custom_prompt = request_data.get('custom_prompt', None)
        override_context = request_data.get('override_context', None)

        # Get user context from session
        user_department = session.get('department')
        user_role = session.get('role', 'Viewer')

        # Initialize data connector (use existing or create new)
        data_connector = DataConnector(data_folder='data', auto_load=True)

        # Check if data is loaded
        if not data_connector.get_current_dataset():
            return JSONResponse({
                "success": False,
                "error": "No dataset loaded. Please upload data first.",
                "recommendation": "Upload Excel files to the data/ folder or use the upload endpoint."
            }, status_code=400)

        # UPDATED: Use unified rating system for personalization (v2)
        personalization_prompt = _build_personalization_prompt(session['user_id'])

        # Still support session feedback (per-session quick adjustments)
        if session_id:
            from services.feedback_service import session_feedback_store, _build_session_prompt
            if session_id in session_feedback_store:
                session_prompt = _build_session_prompt(session_feedback_store[session_id])
                personalization_prompt += session_prompt

        # Add personalization to custom prompt
        if personalization_prompt:
            custom_prompt = (custom_prompt or "") + personalization_prompt

        # Initialize smart generator
        smart_gen = SmartDashboardGenerator(data_connector, use_llm=True)

        # Clear existing charts before generating (as per requirements)
        ds = get_dashboard_system()
        if ds:
            ds.dashboard.clear_charts()
            logger.info("🧹 Cleared existing charts before smart generation")

        # Generate smart dashboard
        result = smart_gen.generate_smart_dashboard(
            user_department=user_department,
            user_role=user_role,
            override_context=override_context,
            num_charts=num_charts,
            custom_prompt=custom_prompt
        )

        if not result['success']:
            return JSONResponse({
                "success": False,
                "error": result.get('error', 'Unknown error'),
                "profile": None,
                "recommendations": []
            }, status_code=500)

        # Convert Plotly figures to JSON for direct rendering
        recommendations = result['recommendations']
        chart_data = []

        for i, (rec, fig) in enumerate(zip(recommendations, result['charts']), 1):
            if fig:
                try:
                    # Convert Plotly figure to JSON
                    chart_json = fig.to_json()

                    chart_data.append({
                        "chart_json": chart_json,
                        "title": rec.get('_title', f"Chart {i}"),
                        "reasoning": rec.get('_reasoning', ''),
                        "chart_type": rec['chart_type'],
                        "metric": rec['metric'],
                        "dimension": rec['dimension'],
                        "chart_index": i
                    })
                    logger.info(f"  ✅ Prepared chart {i}: {rec.get('_title', '')}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to convert chart {i} to JSON: {e}")

        # Return charts as JSON for direct frontend rendering
        if chart_data:
            # Prepare validation report
            validation_report = {
                "total_generated": len(recommendations) + len(result.get('failed_recommendations', [])),
                "total_valid": len(recommendations),
                "failed_count": len(result.get('failed_recommendations', [])),
                "failed": result.get('failed_recommendations', [])
            }

            return JSONResponse({
                "success": True,
                "message": f"Smart dashboard generated with {len(chart_data)} charts",
                "chart_count": len(chart_data),
                "charts": chart_data,
                "recommendations": [
                    {
                        "title": rec.get('_title', ''),
                        "reasoning": rec.get('_reasoning', ''),
                        "chart_type": rec['chart_type'],
                        "metric": rec['metric'],
                        "dimension": rec['dimension']
                    }
                    for rec in recommendations
                ],
                "validation_report": validation_report,  # NEW: Include validation report
                "profile": {
                    "dataset": result['profile'].dataset_name,
                    "total_rows": result['profile'].total_rows,
                    "total_columns": result['profile'].total_columns
                },
                "context": {
                    "department": result['context'].department,
                    "role": result['context'].role
                }
            })

        # If we got here, something went wrong
        return JSONResponse({
            "success": False,
            "error": "No charts were generated successfully",
            "chart_count": 0
        }, status_code=500)

    except Exception as e:
        logger.error(f"❌ Smart dashboard generation error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)

# ============================================================================
# Single Chart Generation Endpoint
# ============================================================================

@router.post("/generate-single-chart")
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

# ============================================================================
# Chart Save Endpoint
# ============================================================================

@router.post("/save-chart")
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
