"""
Add Dashboard table to the database
Run this script to create the new dashboard table
"""
from sqlalchemy import create_engine
from database.models import Base, Dashboard, User

# Database URL - same as main.py
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/analytics_dashboard"

# Create engine
engine = create_engine(DATABASE_URL)

# Create all tables (will only create missing ones)
print("Creating Dashboard table...")
Base.metadata.create_all(bind=engine)
print("✅ Dashboard table created successfully!")
print("\nYou can now use dashboard history features.")
