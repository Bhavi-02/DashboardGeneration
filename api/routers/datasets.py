"""Dataset management API router for Gen-Dash"""
from fastapi import APIRouter, Depends, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any
from api.dependencies import require_auth, get_dashboard_system
import traceback

router = APIRouter()

@router.get("/get-dataset-info")
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
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Error retrieving dataset information: {str(e)}"
        }, status_code=500)

@router.get("/get-available-datasets")
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
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@router.post("/switch-dataset")
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
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)
