"""Export endpoints for Gen-Dash"""
from fastapi import APIRouter, Depends, Body, Path, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import Dict, Any
from pathlib import Path as PathLib
import pandas as pd
import io
import json
import zipfile
import traceback
from datetime import datetime

from api.dependencies import require_auth, get_db, get_dashboard_system
from database.models import Dashboard
from core.database import SessionLocal

router = APIRouter()

# API: Export dashboard data as CSV/Excel/JSON
@router.post("/api/export-dashboard-data")
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
        export_dir = PathLib("exports")
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
                    filename=PathLib(csv_files[0]).name
                )
            else:
                # Multiple files - create zip
                zip_filename = f"{base_filename}.zip"
                zip_path = export_dir / zip_filename

                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for csv_file in csv_files:
                        zipf.write(csv_file, PathLib(csv_file).name)

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
@router.post("/api/export-dataset-tables")
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
        export_dir = PathLib("exports")
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
                    filename=PathLib(csv_files[0]).name
                )
            else:
                zip_filename = f"{base_filename}.zip"
                zip_path = export_dir / zip_filename

                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for csv_file in csv_files:
                        zipf.write(csv_file, PathLib(csv_file).name)

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
@router.get("/api/export-dashboard-html/{dashboard_id}")
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
        dashboard_file = PathLib(dashboard.file_path)
        if not dashboard_file.exists():
            return JSONResponse({
                "success": False,
                "error": "Dashboard file not found"
            }, status_code=404)

        # Create exports directory
        export_dir = PathLib("exports")
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
