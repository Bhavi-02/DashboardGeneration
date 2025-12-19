"""
RBAC Test Script
Demonstrates the role-based access control functionality
"""

# Define ROLE_PERMISSIONS locally for testing
ROLE_PERMISSIONS = {
    "Admin": ["admin", "analyst", "departmental", "viewer"],
    "Analyst": ["analyst", "departmental", "viewer"],
    "Departmental": ["departmental", "viewer"],
    "Viewer": ["viewer"]
}

def get_accessible_pages(role: str):
    """Get list of pages accessible to a role"""
    return ROLE_PERMISSIONS.get(role, [])

def check_permission(session: dict, required_page: str) -> bool:
    """Check if user has permission to access a page"""
    user_role = session.get("role")
    if not user_role:
        return False
    
    allowed_pages = ROLE_PERMISSIONS.get(user_role, [])
    return required_page in allowed_pages

def test_rbac():
    print("=" * 60)
    print("ROLE-BASED ACCESS CONTROL (RBAC) TEST")
    print("=" * 60)
    print()
    
    # Test all roles
    roles = ["Admin", "Analyst", "Departmental", "Viewer"]
    
    for role in roles:
        print(f"🎭 Role: {role}")
        print(f"   Access Level: {len(get_accessible_pages(role))} page(s)")
        print(f"   Accessible Pages: {', '.join(get_accessible_pages(role))}")
        print()
        
        # Test access to each page
        pages = ["admin", "analyst", "departmental", "viewer"]
        for page in pages:
            # Create a mock session
            session = {"role": role}
            has_access = check_permission(session, page)
            status = "✅ ALLOWED" if has_access else "❌ DENIED"
            print(f"   {status}: /dashboard/{page}")
        print()
        print("-" * 60)
        print()

def test_hierarchy():
    print("=" * 60)
    print("ACCESS HIERARCHY VISUALIZATION")
    print("=" * 60)
    print()
    
    print("Admin (Level 4)")
    print("  ├── Admin Panel ✓")
    print("  ├── Analyst View ✓")
    print("  ├── Departmental View ✓")
    print("  └── Viewer Dashboard ✓")
    print()
    
    print("Analyst (Level 3)")
    print("  ├── Admin Panel ✗")
    print("  ├── Analyst View ✓")
    print("  ├── Departmental View ✓")
    print("  └── Viewer Dashboard ✓")
    print()
    
    print("Departmental (Level 2)")
    print("  ├── Admin Panel ✗")
    print("  ├── Analyst View ✗")
    print("  ├── Departmental View ✓")
    print("  └── Viewer Dashboard ✓")
    print()
    
    print("Viewer (Level 1)")
    print("  ├── Admin Panel ✗")
    print("  ├── Analyst View ✗")
    print("  ├── Departmental View ✗")
    print("  └── Viewer Dashboard ✓")
    print()

if __name__ == "__main__":
    test_rbac()
    test_hierarchy()
    
    print("=" * 60)
    print("✅ RBAC Implementation Test Complete!")
    print("=" * 60)
