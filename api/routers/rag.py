"""RAG (Retrieval-Augmented Generation) API router for Gen-Dash"""
from fastapi import APIRouter, Depends, Body, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from api.dependencies import require_auth
import traceback

router = APIRouter()

@router.post("/upload-file-for-rag")
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

@router.post("/process-rag-document")
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

@router.post("/query-rag")
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
