"""
Migration script to add is_ai_generated column to dashboards table
Run this ONCE to update existing database schema

Usage:
    python database/add_ai_generated_column.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import create_engine, text
from database.models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:dhruv123@localhost:3306/analytics_dashboard")

def add_ai_generated_column():
    """Add is_ai_generated column to dashboards table"""
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT COUNT(*) as count
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = 'analytics_dashboard'
                AND TABLE_NAME = 'dashboards'
                AND COLUMN_NAME = 'is_ai_generated'
            """))
            
            exists = result.fetchone()[0] > 0
            
            if exists:
                logger.info("✅ Column 'is_ai_generated' already exists. No migration needed.")
                return
            
            # Add column
            logger.info("📝 Adding 'is_ai_generated' column to dashboards table...")
            conn.execute(text("""
                ALTER TABLE dashboards
                ADD COLUMN is_ai_generated BOOLEAN DEFAULT FALSE
            """))
            conn.commit()
            
            logger.info("✅ Migration complete! Column 'is_ai_generated' added successfully.")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    add_ai_generated_column()
