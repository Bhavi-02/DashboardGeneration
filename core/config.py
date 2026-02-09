"""Core configuration settings for Gen-Dash"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Database configuration
DATABASE_URL = "mysql+pymysql://root:dhruv123@localhost:3306/analytics_dashboard"

def setup_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/gendash.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def setup_cors(app: FastAPI):
    """Add CORS middleware to allow frontend requests"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
