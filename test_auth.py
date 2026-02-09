"""
Test Authentication Flow
=========================

This script tests if the authentication system is working correctly.
"""

from auth import (
    get_supabase_client,
    create_user,
    authenticate_user,
    user_exists
)

def test_supabase_connection():
    """Test Supabase connection"""
    print("Testing Supabase connection...")
    client = get_supabase_client()
    if client:
        print("✅ Supabase connection successful!")
        return True
    else:
        print("❌ Supabase connection failed!")
        return False

def test_user_operations():
    """Test user operations"""
    test_email = "test@wellio.com"
    
    print(f"\nChecking if test user exists: {test_email}")
    exists = user_exists(test_email)
    print(f"User exists: {exists}")
    
    if not exists:
        print("\nCreating test user...")
        success, message = create_user(test_email, "Test1234", "Test User", "en")
        print(f"Create user result: {message}")
    
    print("\nTesting authentication...")
    success, user, message = authenticate_user(test_email, "Test1234")
    if success:
        print(f"✅ Authentication successful!")
        print(f"   User: {user.name}")
        print(f"   Email: {user.email}")
        print(f"   Language: {user.language}")
    else:
        print(f"❌ Authentication failed: {message}")

if __name__ == "__main__":
    print("=" * 60)
    print("WELLIO AUTHENTICATION TEST")
    print("=" * 60)
    
    if test_supabase_connection():
        test_user_operations()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
