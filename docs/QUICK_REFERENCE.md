# Quick Reference: Logging & Validation

## Logging

### Import Loggers

```python
from utils.logger import (
    app_logger,      # General application logs
    db_logger,       # Database operations
    auth_logger,     # Authentication
    chart_logger,    # Chart generation
    dashboard_logger,# Dashboard creation
    nlu_logger,      # NLP operations
    rag_logger       # RAG system
)
```

### Usage Examples

```python
# Info level (general operations)
app_logger.info("Server started successfully")
chart_logger.info(f"Generated chart: {chart_type}")

# Warning level (recoverable issues)
app_logger.warning("Connection pool near capacity")
db_logger.warning(f"Slow query detected: {query_time}ms")

# Error level (errors that need attention)
auth_logger.error(f"Failed login attempt for user: {username}")
chart_logger.error(f"Chart generation failed: {error}")

# Debug level (detailed troubleshooting)
nlu_logger.debug(f"Extracted entities: {entities}")
rag_logger.debug(f"Vector search results: {results}")
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: Confirmation that things are working
- **WARNING**: Indication of potential problems
- **ERROR**: Serious problem occurred
- **CRITICAL**: Very serious error

---

## Input Validation

### Import Validators

```python
from utils.validators import (
    sanitize_string,
    validate_email,
    validate_username,
    validate_password,
    validate_role,
    sanitize_filename,
    validate_integer
)
```

### String Sanitization

```python
# Sanitize user input (XSS prevention)
safe_title = sanitize_string(user_input, max_length=200)

# Result: HTML escaped, trimmed, length-limited
```

### Email Validation

```python
is_valid = validate_email("user@example.com")
# Returns: True or False
```

### Username Validation

```python
is_valid, error_msg = validate_username("john_doe")
# Returns: (True, None) or (False, "error message")

if not is_valid:
    return {"error": error_msg}
```

### Password Validation

```python
is_valid, error_msg = validate_password("SecurePass123")
# Returns: (True, None) or (False, "error message")

# Checks:
# - Minimum 8 characters
# - Maximum 72 characters (bcrypt limit)
```

### Role Validation

```python
is_valid = validate_role("Admin")
# Returns: True if role in ["Admin", "Analyst", "Departmental", "Viewer"]
```

### Filename Sanitization

```python
safe_filename = sanitize_filename("../../etc/passwd")
# Returns: "______etc_passwd" (directory traversal prevented)
```

### Integer Validation

```python
is_valid, value, error_msg = validate_integer("42", min_val=1, max_val=100)
# Returns: (True, 42, None) or (False, None, "error message")

if is_valid:
    process_value(value)
else:
    return {"error": error_msg}
```

---

## Complete Endpoint Example

```python
from fastapi import FastAPI, HTTPException, Form, Depends
from utils.logger import auth_logger
from utils.validators import (
    validate_username,
    validate_password,
    validate_email,
    sanitize_string
)

@app.post("/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    full_name: str = Form(...)
):
    """Register new user with validation"""

    # 1. Validate username
    is_valid, error = validate_username(username)
    if not is_valid:
        auth_logger.warning(f"Invalid username attempt: {username}")
        raise HTTPException(status_code=400, detail=error)

    # 2. Validate password
    is_valid, error = validate_password(password)
    if not is_valid:
        auth_logger.warning(f"Weak password for user: {username}")
        raise HTTPException(status_code=400, detail=error)

    # 3. Validate email
    if not validate_email(email):
        auth_logger.warning(f"Invalid email: {email}")
        raise HTTPException(status_code=400, detail="Invalid email format")

    # 4. Sanitize full name
    full_name = sanitize_string(full_name, max_length=100)

    # 5. Create user
    try:
        # Database operations...
        auth_logger.info(f"New user registered: {username}")
        return {"success": True, "message": "User created"}
    except Exception as e:
        auth_logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
```

---

## Database Session Management

### Use Dependency Injection

```python
from sqlalchemy.orm import Session
from fastapi import Depends

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users")
async def get_users(db: Session = Depends(get_db)):
    """Automatic session cleanup"""
    users = db.query(User).all()
    return users
    # db.close() called automatically
```

### Benefits

- ✅ Automatic connection cleanup
- ✅ Connection pooling (5 connections)
- ✅ Connection recycling (every hour)
- ✅ Health checks before queries
- ✅ No resource leaks

---

## Error Handling Best Practices

### Bad ❌

```python
try:
    result = risky_operation()
except:
    pass
```

### Good ✅

```python
try:
    result = risky_operation()
except ValueError as e:
    logger.warning(f"Invalid value: {e}")
    return default_value
except KeyError as e:
    logger.error(f"Missing key: {e}")
    raise HTTPException(status_code=400, detail="Invalid request")
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

---

## Testing Your Changes

### Test Logging

```python
# Run your application
python main.py

# Check logs
cat logs/gendash.log
cat logs/gendash_auth.log
cat logs/gendash_charts.log
```

### Test Validation

```python
# Test in Python shell
from utils.validators import *

# Test email
print(validate_email("test@example.com"))  # True
print(validate_email("invalid"))            # False

# Test username
print(validate_username("user123"))         # (True, None)
print(validate_username("ab"))              # (False, "too short")

# Test filename sanitization
print(sanitize_filename("../../etc/passwd"))  # "______etc_passwd"
```

---

## Configuration

### Adjust Log Levels

Edit `utils/logger.py`:

```python
# Development (verbose)
setup_logger('gendash', logging.DEBUG)

# Production (minimal)
setup_logger('gendash', logging.WARNING)
```

### Adjust Connection Pool

Edit `main.py`:

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,        # More connections
    max_overflow=20,     # Higher burst capacity
    pool_recycle=1800,   # Recycle every 30 minutes
)
```

---

## Troubleshooting

### Logs Not Appearing

1. Check `logs/` directory exists
2. Verify file permissions
3. Check disk space
4. Review logging configuration

### Connection Pool Exhausted

1. Increase `pool_size`
2. Increase `max_overflow`
3. Check for connection leaks
4. Monitor with: `db.execute("SHOW PROCESSLIST")`

### Validation Too Strict

1. Adjust regex patterns in `validators.py`
2. Modify min/max length constraints
3. Update allowed characters

---

**For detailed documentation, see [OPTIMIZATIONS.md](OPTIMIZATIONS.md)**
