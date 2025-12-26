<<<<<<< HEAD
"""
Script to add department column to existing users table in MySQL
Run this once to update your database schema
"""
from sqlalchemy import create_engine, text

# Your MySQL connection string (updated to match main.py)
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/analytics_dashboard"

def add_department_column():
    engine = create_engine(DATABASE_URL)
    
    try:
        with engine.connect() as connection:
            # Add department column if it doesn't exist
            alter_query = text("""
                ALTER TABLE users 
                ADD COLUMN department VARCHAR(50) NULL AFTER role
            """)
            
            connection.execute(alter_query)
            connection.commit()
            
            print("✅ Successfully added 'department' column to users table")
            
    except Exception as e:
        if "Duplicate column name" in str(e):
            print("ℹ️ Column 'department' already exists")
        else:
            print(f"❌ Error: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    add_department_column()
=======
"""
Script to add department column to existing users table in MySQL
Run this once to update your database schema
"""
from sqlalchemy import create_engine, text

# Your MySQL connection string (updated to match main.py)
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/analytics_dashboard"

def add_department_column():
    engine = create_engine(DATABASE_URL)
    
    try:
        with engine.connect() as connection:
            # Add department column if it doesn't exist
            alter_query = text("""
                ALTER TABLE users 
                ADD COLUMN department VARCHAR(50) NULL AFTER role
            """)
            
            connection.execute(alter_query)
            connection.commit()
            
            print("✅ Successfully added 'department' column to users table")
            
    except Exception as e:
        if "Duplicate column name" in str(e):
            print("ℹ️ Column 'department' already exists")
        else:
            print(f"❌ Error: {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    add_department_column()
>>>>>>> chart-creator
