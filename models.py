from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    department = Column(String(50), nullable=True)  # For departmental users
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}', role='{self.role}')>"

class Dashboard(Base):
    __tablename__ = "dashboards"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    charts_config = Column(Text, nullable=False)  # JSON string of chart queries and configurations
    file_path = Column(String(500), nullable=True)  # Path to generated HTML file
    chart_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    # RBAC visibility controls
    created_by_role = Column(String(50), nullable=True)  # Role of the creator
    visible_to_viewer = Column(Boolean, default=False)  # Can viewers see this dashboard?
    allowed_departments = Column(Text, nullable=True)  # Comma-separated list of departments
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Dashboard(id={self.id}, title='{self.title}', user_id={self.user_id}, charts={self.chart_count})>"