<<<<<<< HEAD
from pydantic import BaseModel, EmailStr, field_validator, model_validator
from typing import Optional

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    username: str
    password: str
    role: str
    department: Optional[str] = None
    
    @field_validator('full_name')
    @classmethod
    def validate_full_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Full name must be at least 2 characters')
        return v.strip()
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.replace('_', '').replace('.', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, underscores, and dots')
        return v.lower()
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password is too long (max 72 bytes)')
        return v
    
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        allowed_roles = ['Admin', 'Analyst', 'Departmental', 'Viewer']
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v
    
    @model_validator(mode='after')
    def validate_department_for_role(self):
        if self.role == 'Departmental' and not self.department:
            raise ValueError('Department is required for departmental users')
        if self.department:
            allowed_departments = ['Sales', 'Marketing', 'Finance', 'Human Resources', 'IT', 'Operations']
            if self.department not in allowed_departments:
                raise ValueError(f'Department must be one of: {", ".join(allowed_departments)}')
        return self

class UserLogin(BaseModel):
    loginId: str  # Can be username or email
    password: str
    role: str
    department: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    username: str
    role: str
    department: Optional[str]
    is_active: bool
    
=======
from pydantic import BaseModel, EmailStr, field_validator, model_validator
from typing import Optional

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    username: str
    password: str
    role: str
    department: Optional[str] = None
    
    @field_validator('full_name')
    @classmethod
    def validate_full_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Full name must be at least 2 characters')
        return v.strip()
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.replace('_', '').replace('.', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, underscores, and dots')
        return v.lower()
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password is too long (max 72 bytes)')
        return v
    
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        allowed_roles = ['Admin', 'Analyst', 'Departmental', 'Viewer']
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v
    
    @model_validator(mode='after')
    def validate_department_for_role(self):
        if self.role == 'Departmental' and not self.department:
            raise ValueError('Department is required for departmental users')
        if self.department:
            allowed_departments = ['Sales', 'Marketing', 'Finance', 'Human Resources', 'IT', 'Operations']
            if self.department not in allowed_departments:
                raise ValueError(f'Department must be one of: {", ".join(allowed_departments)}')
        return self

class UserLogin(BaseModel):
    loginId: str  # Can be username or email
    password: str
    role: str
    department: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    username: str
    role: str
    department: Optional[str]
    is_active: bool
    
>>>>>>> chart-creator
    model_config = {"from_attributes": True}