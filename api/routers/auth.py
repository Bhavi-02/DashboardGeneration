"""Authentication router for user registration and login"""
from fastapi import APIRouter, Depends, Form, Request, Response, Cookie, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional

# Import database dependencies
from core.database import SessionLocal
from database.models import User
from database.schemas import UserCreate

# Import auth dependencies
from api.dependencies import (
    require_auth,
    get_session,
    create_session,
    get_password_hash,
    verify_password,
    get_accessible_pages
)

router = APIRouter()

# Session info endpoint
@router.get("/api/session-info")
async def get_session_info(session: dict = Depends(require_auth)):
    """Return current user's session information"""
    return JSONResponse({
        "user_id": session["user_id"],
        "username": session["username"],
        "email": session["email"],
        "full_name": session["full_name"],
        "role": session["role"],
        "department": session.get("department")
    })

# Check session endpoint (for frontend validation)
@router.get("/api/check-session")
async def check_session(session_id: Optional[str] = Cookie(None)):
    """Check if user has a valid session"""
    session = get_session(session_id)
    if session:
        return JSONResponse({
            "authenticated": True,
            "user": session["username"],
            "role": session["role"]
        })
    return JSONResponse({
        "authenticated": False
    })

@router.post("/register")
async def register(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    department: Optional[str] = Form(None)
):
    db = SessionLocal()
    try:
        user_data = UserCreate(
            full_name=full_name,
            email=email,
            username=username,
            password=password,
            role=role,
            department=department
        )

        existing = db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        if existing:
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Registration Failed</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                        margin: 5px;
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1>Registration Failed!</h1>
                    <p>Username or email already exists. Please try with different credentials.</p>
                    <a href="/register.html" class="btn">Try Again</a>
                    <a href="/page.html" class="btn" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Login Instead</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=400)

        db_user = User(
            full_name=user_data.full_name,
            email=user_data.email,
            username=user_data.username,
            hashed_password=get_password_hash(user_data.password),
            role=user_data.role,
            department=user_data.department
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        success_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Registration Successful</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .success-container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 600px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(-50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .success-icon {{
                    font-size: 100px;
                    margin-bottom: 20px;
                    animation: bounce 1s ease-in-out;
                }}
                @keyframes bounce {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.2); }}
                }}
                h1 {{
                    color: #27ae60;
                    margin-bottom: 15px;
                    font-size: 32px;
                }}
                .subtitle {{
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 18px;
                }}
                .user-info {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    text-align: left;
                }}
                .user-info h3 {{
                    color: #333;
                    margin-bottom: 15px;
                    font-size: 20px;
                    text-align: center;
                }}
                .info-row {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #ddd;
                }}
                .info-row:last-child {{
                    border-bottom: none;
                }}
                .info-label {{
                    font-weight: 600;
                    color: #555;
                }}
                .info-value {{
                    color: #333;
                }}
                .success-message {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    font-size: 16px;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    margin: 5px;
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="success-container">
                <div class="success-icon">🎉</div>
                <h1>Registration Successful!</h1>
                <p class="subtitle">Your account has been created successfully!</p>

                <div class="success-message">
                    ✅ Data has been saved to MySQL database
                </div>

                <div class="user-info">
                    <h3>📋 Your Account Details</h3>
                    <div class="info-row">
                        <span class="info-label">👤 Full Name:</span>
                        <span class="info-value">{db_user.full_name}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">🆔 Username:</span>
                        <span class="info-value">{db_user.username}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">📧 Email:</span>
                        <span class="info-value">{db_user.email}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">🎭 Role:</span>
                        <span class="info-value">{db_user.role}</span>
                    </div>
                    {f'<div class="info-row"><span class="info-label">🏢 Department:</span><span class="info-value">{db_user.department}</span></div>' if db_user.department else ''}
                </div>

                <a href="/page.html" class="btn">Login Now</a>
                <a href="/register.html" class="btn" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">Register Another</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=success_html, status_code=200)

    except ValueError as e:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Validation Error</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .error-container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 500px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(-50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .error-icon {{
                    font-size: 80px;
                    margin-bottom: 20px;
                    animation: shake 0.5s ease-in-out;
                }}
                @keyframes shake {{
                    0%, 100% {{ transform: translateX(0); }}
                    25% {{ transform: translateX(-10px); }}
                    75% {{ transform: translateX(10px); }}
                }}
                h1 {{
                    color: #e74c3c;
                    margin-bottom: 15px;
                    font-size: 28px;
                }}
                p {{
                    color: #555;
                    margin-bottom: 30px;
                    font-size: 16px;
                    line-height: 1.6;
                }}
                .error-detail {{
                    background: #fff3cd;
                    border: 1px solid #ffc107;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    color: #856404;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <h1>Validation Error!</h1>
                <p>Please check the following error and try again:</p>
                <div class="error-detail">
                    {str(e)}
                </div>
                <a href="/register.html" class="btn">Try Again</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=422)
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Registration Error</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .error-container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 500px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(-50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .error-icon {{
                    font-size: 80px;
                    margin-bottom: 20px;
                }}
                h1 {{
                    color: #e74c3c;
                    margin-bottom: 15px;
                    font-size: 28px;
                }}
                p {{
                    color: #555;
                    margin-bottom: 30px;
                    font-size: 16px;
                    line-height: 1.6;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <div class="error-icon">❌</div>
                <h1>Registration Failed!</h1>
                <p>An unexpected error occurred. Please try again.</p>
                <a href="/register.html" class="btn">Try Again</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)
    finally:
        db.close()

@router.post("/login")
async def login(
    response: Response,
    request: Request,
    loginId: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    department: Optional[str] = Form(None)
):
    db = SessionLocal()
    try:
        # Find user by username or email
        db_user = db.query(User).filter(
            (User.username == loginId.lower()) |
            (User.email == loginId.lower())
        ).first()

        if not db_user:
            # Return HTML error page
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Login Failed</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1>Login Failed!</h1>
                    <p>Details not matched. Please check your credentials and try again.</p>
                    <a href="/page.html" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=401)

        if not verify_password(password, db_user.hashed_password):
            # Return HTML error page
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Login Failed</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1>Login Failed!</h1>
                    <p>Details not matched. Please check your credentials and try again.</p>
                    <a href="/page.html" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=401)

        if not db_user.is_active:
            # Return HTML error page
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Account Inactive</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">🚫</div>
                    <h1>Account Inactive</h1>
                    <p>Your account is currently inactive. Please contact support.</p>
                    <a href="/page.html" class="btn">Back to Login</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=401)

        # Verify role matches
        if db_user.role != role:
            # Return HTML error page
            error_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Role Mismatch</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .error-container {
                        background: white;
                        border-radius: 20px;
                        padding: 40px;
                        max-width: 500px;
                        width: 100%;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                        text-align: center;
                        animation: slideIn 0.5s ease-out;
                    }
                    @keyframes slideIn {
                        from { transform: translateY(-50px); opacity: 0; }
                        to { transform: translateY(0); opacity: 1; }
                    }
                    .error-icon {
                        font-size: 80px;
                        margin-bottom: 20px;
                        animation: shake 0.5s ease-in-out;
                    }
                    @keyframes shake {
                        0%, 100% { transform: translateX(0); }
                        25% { transform: translateX(-10px); }
                        75% { transform: translateX(10px); }
                    }
                    h1 {
                        color: #e74c3c;
                        margin-bottom: 15px;
                        font-size: 28px;
                    }
                    p {
                        color: #555;
                        margin-bottom: 30px;
                        font-size: 16px;
                        line-height: 1.6;
                    }
                    .btn {
                        display: inline-block;
                        padding: 12px 40px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 30px;
                        font-weight: 600;
                        transition: transform 0.3s, box-shadow 0.3s;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    .btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <h1>Login Failed!</h1>
                    <p>Role mismatch. Details not matched. Please try again.</p>
                    <a href="/page.html" class="btn">Try Again</a>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html, status_code=401)

        # For departmental users, verify department matches
        if role == "Departmental":
            if not department or db_user.department != department:
                # Return HTML error page
                error_html = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Department Mismatch</title>
                    <style>
                        * { margin: 0; padding: 0; box-sizing: border-box; }
                        body {
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            min-height: 100vh;
                            padding: 20px;
                        }
                        .error-container {
                            background: white;
                            border-radius: 20px;
                            padding: 40px;
                            max-width: 500px;
                            width: 100%;
                            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                            text-align: center;
                            animation: slideIn 0.5s ease-out;
                        }
                        @keyframes slideIn {
                            from { transform: translateY(-50px); opacity: 0; }
                            to { transform: translateY(0); opacity: 1; }
                        }
                        .error-icon {
                            font-size: 80px;
                            margin-bottom: 20px;
                            animation: shake 0.5s ease-in-out;
                        }
                        @keyframes shake {
                            0%, 100% { transform: translateX(0); }
                            25% { transform: translateX(-10px); }
                            75% { transform: translateX(10px); }
                        }
                        h1 {
                            color: #e74c3c;
                            margin-bottom: 15px;
                            font-size: 28px;
                        }
                        p {
                            color: #555;
                            margin-bottom: 30px;
                            font-size: 16px;
                            line-height: 1.6;
                        }
                        .btn {
                            display: inline-block;
                            padding: 12px 40px;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            text-decoration: none;
                            border-radius: 30px;
                            font-weight: 600;
                            transition: transform 0.3s, box-shadow 0.3s;
                            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                        }
                        .btn:hover {
                            transform: translateY(-2px);
                            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                        }
                    </style>
                </head>
                <body>
                    <div class="error-container">
                        <div class="error-icon">❌</div>
                        <h1>Login Failed!</h1>
                        <p>Department mismatch. Details not matched. Please try again.</p>
                        <a href="/page.html" class="btn">Try Again</a>
                    </div>
                </body>
                </html>
                """
                return HTMLResponse(content=error_html, status_code=401)

        # If all validations pass, create session and redirect to home
        user_data = {
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email,
            "full_name": db_user.full_name,
            "role": db_user.role,
            "department": db_user.department
        }

        session_id = create_session(user_data)

        # Get accessible pages for this role
        accessible_pages = get_accessible_pages(db_user.role)
        pages_list = ", ".join(accessible_pages)

        success_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Login Successful</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    padding: 20px;
                }}
                .success-container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 600px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    text-align: center;
                    animation: slideIn 0.5s ease-out;
                }}
                @keyframes slideIn {{
                    from {{ transform: translateY(-50px); opacity: 0; }}
                    to {{ transform: translateY(0); opacity: 1; }}
                }}
                .success-icon {{
                    font-size: 100px;
                    margin-bottom: 20px;
                    animation: bounce 1s ease-in-out;
                }}
                @keyframes bounce {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.2); }}
                }}
                h1 {{
                    color: #27ae60;
                    margin-bottom: 15px;
                    font-size: 32px;
                }}
                .subtitle {{
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 18px;
                }}
                .user-info {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    text-align: left;
                }}
                .user-info h3 {{
                    color: #333;
                    margin-bottom: 15px;
                    font-size: 20px;
                }}
                .info-row {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #ddd;
                }}
                .info-row:last-child {{
                    border-bottom: none;
                }}
                .info-label {{
                    font-weight: 600;
                    color: #555;
                }}
                .info-value {{
                    color: #333;
                }}
                .access-info {{
                    background: #e8f5e9;
                    border-left: 4px solid #4caf50;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    text-align: left;
                }}
                .access-info h4 {{
                    color: #2e7d32;
                    margin-bottom: 10px;
                }}
                .access-info p {{
                    color: #555;
                    margin: 5px 0;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 30px;
                    font-weight: 600;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    margin: 5px;
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                }}
            </style>
        </head>
        <body>
            <div class="success-container">
                <div class="success-icon">✅</div>
                <h1>Details Validated Great!</h1>
                <p class="subtitle">Welcome back, {db_user.full_name}!</p>

                <div class="user-info">
                    <h3>📋 Account Information</h3>
                    <div class="info-row">
                        <span class="info-label">👤 Username:</span>
                        <span class="info-value">{db_user.username}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">📧 Email:</span>
                        <span class="info-value">{db_user.email}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">🎭 Role:</span>
                        <span class="info-value">{db_user.role}</span>
                    </div>
                    {f'<div class="info-row"><span class="info-label">🏢 Department:</span><span class="info-value">{db_user.department}</span></div>' if db_user.department else ''}
                </div>

                <div class="access-info">
                    <h4>🔐 Your Access Permissions (RBAC)</h4>
                    <p><strong>You can access {len(accessible_pages)} page(s):</strong></p>
                    <p>✓ {pages_list}</p>
                    <p style="margin-top: 10px; font-size: 14px; color: #666;">
                        Role-Based Access Control ensures you only see pages relevant to your role,
                        protecting sensitive data and maintaining privacy.
                    </p>
                </div>

                <a href="/home" class="btn">Go to Dashboard</a>
            </div>
        </body>
        </html>
        """

        response = HTMLResponse(content=success_html, status_code=200)
        response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=1800)  # 30 min
        return response
    finally:
        db.close()
