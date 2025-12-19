"""
Input validation and sanitization utilities
"""
import re
from typing import Optional
from html import escape

def sanitize_string(text: str, max_length: int = 500) -> str:
    """
    Sanitize user input string
    
    Args:
        text: Input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not text:
        return ""
    
    # Trim and limit length
    text = str(text).strip()[:max_length]
    
    # HTML escape to prevent XSS
    text = escape(text)
    
    return text

def validate_email(email: str) -> bool:
    """
    Validate email format
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_username(username: str) -> tuple[bool, Optional[str]]:
    """
    Validate username format
    
    Args:
        username: Username to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not username:
        return False, "Username is required"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if len(username) > 50:
        return False, "Username must be less than 50 characters"
    
    # Only allow alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return False, "Username can only contain letters, numbers, underscore, and hyphen"
    
    return True, None

def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password strength
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, "Password is required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if len(password) > 72:  # bcrypt limit
        return False, "Password must be less than 72 characters"
    
    # Check for complexity (optional - can be enabled)
    # has_upper = bool(re.search(r'[A-Z]', password))
    # has_lower = bool(re.search(r'[a-z]', password))
    # has_digit = bool(re.search(r'\d', password))
    # has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    # 
    # if not (has_upper and has_lower and has_digit):
    #     return False, "Password must contain uppercase, lowercase, and numbers"
    
    return True, None

def validate_role(role: str) -> bool:
    """
    Validate user role
    
    Args:
        role: Role to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_roles = ["Admin", "Analyst", "Departmental", "Viewer"]
    return role in valid_roles

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal attacks
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed"
    
    # Remove path separators and dangerous characters
    filename = re.sub(r'[/\\<>:"|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    
    return filename or "unnamed"

def validate_integer(value: any, min_val: int = None, max_val: int = None) -> tuple[bool, Optional[int], Optional[str]]:
    """
    Validate and convert to integer
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Tuple of (is_valid, converted_value, error_message)
    """
    try:
        int_val = int(value)
        
        if min_val is not None and int_val < min_val:
            return False, None, f"Value must be at least {min_val}"
        
        if max_val is not None and int_val > max_val:
            return False, None, f"Value must be at most {max_val}"
        
        return True, int_val, None
        
    except (ValueError, TypeError):
        return False, None, "Invalid integer value"
