<<<<<<< HEAD
"""
Add visibility control columns to dashboards table for RBAC
"""
from sqlalchemy import create_engine, text

# Database connection
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/analytics_dashboard"
engine = create_engine(DATABASE_URL)

def add_visibility_columns():
    """Add new columns for dashboard visibility control"""
    
    with engine.connect() as conn:
        try:
            # Add visible_to_viewer column (boolean)
            print("Adding visible_to_viewer column...")
            conn.execute(text("""
                ALTER TABLE dashboards 
                ADD COLUMN visible_to_viewer BOOLEAN DEFAULT FALSE
            """))
            conn.commit()
            print("✅ visible_to_viewer column added")
            
        except Exception as e:
            if "Duplicate column name" in str(e):
                print("⚠️ visible_to_viewer column already exists")
            else:
                print(f"❌ Error adding visible_to_viewer: {e}")
        
        try:
            # Add allowed_departments column (comma-separated string)
            print("Adding allowed_departments column...")
            conn.execute(text("""
                ALTER TABLE dashboards 
                ADD COLUMN allowed_departments TEXT
            """))
            conn.commit()
            print("✅ allowed_departments column added")
            
        except Exception as e:
            if "Duplicate column name" in str(e):
                print("⚠️ allowed_departments column already exists")
            else:
                print(f"❌ Error adding allowed_departments: {e}")
        
        try:
            # Add created_by_role column to track creator's role
            print("Adding created_by_role column...")
            conn.execute(text("""
                ALTER TABLE dashboards 
                ADD COLUMN created_by_role VARCHAR(50)
            """))
            conn.commit()
            print("✅ created_by_role column added")
            
        except Exception as e:
            if "Duplicate column name" in str(e):
                print("⚠️ created_by_role column already exists")
            else:
                print(f"❌ Error adding created_by_role: {e}")
    
    print("\n🎉 Database migration completed!")
    print("\nNew columns added:")
    print("  - visible_to_viewer: BOOLEAN (default FALSE)")
    print("  - allowed_departments: TEXT (comma-separated department names)")
    print("  - created_by_role: VARCHAR(50) (role of dashboard creator)")

if __name__ == "__main__":
    print("🔧 Starting database migration...")
    print(f"📦 Database: {DATABASE_URL}")
    print()
    add_visibility_columns()
=======
"""
Add visibility control columns to dashboards table for RBAC
"""
from sqlalchemy import create_engine, text

# Database connection
DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/analytics_dashboard"
engine = create_engine(DATABASE_URL)

def add_visibility_columns():
    """Add new columns for dashboard visibility control"""
    
    with engine.connect() as conn:
        try:
            # Add visible_to_viewer column (boolean)
            print("Adding visible_to_viewer column...")
            conn.execute(text("""
                ALTER TABLE dashboards 
                ADD COLUMN visible_to_viewer BOOLEAN DEFAULT FALSE
            """))
            conn.commit()
            print("✅ visible_to_viewer column added")
            
        except Exception as e:
            if "Duplicate column name" in str(e):
                print("⚠️ visible_to_viewer column already exists")
            else:
                print(f"❌ Error adding visible_to_viewer: {e}")
        
        try:
            # Add allowed_departments column (comma-separated string)
            print("Adding allowed_departments column...")
            conn.execute(text("""
                ALTER TABLE dashboards 
                ADD COLUMN allowed_departments TEXT
            """))
            conn.commit()
            print("✅ allowed_departments column added")
            
        except Exception as e:
            if "Duplicate column name" in str(e):
                print("⚠️ allowed_departments column already exists")
            else:
                print(f"❌ Error adding allowed_departments: {e}")
        
        try:
            # Add created_by_role column to track creator's role
            print("Adding created_by_role column...")
            conn.execute(text("""
                ALTER TABLE dashboards 
                ADD COLUMN created_by_role VARCHAR(50)
            """))
            conn.commit()
            print("✅ created_by_role column added")
            
        except Exception as e:
            if "Duplicate column name" in str(e):
                print("⚠️ created_by_role column already exists")
            else:
                print(f"❌ Error adding created_by_role: {e}")
    
    print("\n🎉 Database migration completed!")
    print("\nNew columns added:")
    print("  - visible_to_viewer: BOOLEAN (default FALSE)")
    print("  - allowed_departments: TEXT (comma-separated department names)")
    print("  - created_by_role: VARCHAR(50) (role of dashboard creator)")

if __name__ == "__main__":
    print("🔧 Starting database migration...")
    print(f"📦 Database: {DATABASE_URL}")
    print()
    add_visibility_columns()
>>>>>>> chart-creator
