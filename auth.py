"""
User Authentication Module with Supabase
=========================================

Secure user authentication with bcrypt password hashing and Supabase cloud database
for the Wellio health monitoring application.

Security Features:
- bcrypt password hashing with automatic salting
- Supabase PostgreSQL cloud database
- Email validation
- Password strength validation
- Duplicate email prevention
- Secure cloud storage
"""

import os
import re
import json
import base64
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import requests
import jwt
import bcrypt
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# Ensure env vars are loaded module-level
load_dotenv(override=True)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class User:
    """User account data"""
    email: str
    password_hash: str
    name: str
    language: str = "en"
    created_at: str = ""
    last_login: Optional[str] = None


# ============================================================================
# SUPABASE CONFIGURATION
# ============================================================================

def get_supabase_client() -> Optional[Client]:
    """
    Get Supabase client instance.
    
    Returns:
        Supabase client or None if credentials not found
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("Warning: Supabase credentials not found in environment variables")
        return None
    
    try:
        return create_client(url, key)
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        return None


# ============================================================================
# PASSWORD HASHING
# ============================================================================

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt with automatic salt generation.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify password against stored hash.
    
    Args:
        password: Plain text password to verify
        password_hash: Stored bcrypt hash
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False


# ============================================================================
# VALIDATION
# ============================================================================

def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        (is_valid: bool, error_message: str)
    """
    if not email:
        return False, "Email is required"
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Please enter a valid email address"
    
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password strength.
    
    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    
    Args:
        password: Password to validate
        
    Returns:
        (is_valid: bool, error_message: str)
    """
    if not password:
        return False, "Password is required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    
    return True, ""


def get_password_strength(password: str) -> str:
    """
    Get password strength indicator.
    
    Args:
        password: Password to evaluate
        
    Returns:
        "Weak", "Medium", or "Strong"
    """
    score = 0
    
    if len(password) >= 8:
        score += 1
    if len(password) >= 12:
        score += 1
    if re.search(r'[A-Z]', password):
        score += 1
    if re.search(r'[a-z]', password):
        score += 1
    if re.search(r'[0-9]', password):
        score += 1
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1
    
    if score <= 2:
        return "Weak"
    elif score <= 4:
        return "Medium"
    else:
        return "Strong"


# ============================================================================
# USER MANAGEMENT (SUPABASE)
# ============================================================================

def user_exists(email: str) -> bool:
    """
    Check if user with email exists in Supabase.
    
    Args:
        email: Email address to check
        
    Returns:
        True if user exists, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        response = supabase.table('users').select('email').eq('email', email.lower()).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"Error checking user existence: {e}")
        return False


def get_user(email: str) -> Optional[User]:
    """
    Get user from Supabase by email.
    
    Args:
        email: Email address
        
    Returns:
        User object if found, None otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return None
        
        response = supabase.table('users').select('*').eq('email', email.lower()).execute()
        
        if not response.data:
            return None
        
        user_data = response.data[0]
        return User(
            email=user_data['email'],
            password_hash=user_data['password_hash'],
            name=user_data['name'],
            language=user_data.get('language', 'en'),
            created_at=user_data['created_at'],
            last_login=user_data.get('last_login')
        )
    except Exception as e:
        print(f"Error getting user: {e}")
        return None


def create_user(email: str, password: str, name: str, language: str = "en") -> Tuple[bool, str]:
    """
    Create new user account in Supabase.
    
    Args:
        email: User email
        password: Plain text password
        name: User full name
        language: Preferred language (en, hi, te)
        
    Returns:
        (success: bool, message: str)
    """
    # Validate email
    valid_email, email_msg = validate_email(email)
    if not valid_email:
        return False, email_msg
    
    # Validate password
    valid_pass, pass_msg = validate_password(password)
    if not valid_pass:
        return False, pass_msg
    
    # Validate name
    if not name or len(name.strip()) < 2:
        return False, "Please enter your full name"
    
    # Check credentials first
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        return False, "Missing Supabase configuration. Check .env file."

    try:
        supabase = get_supabase_client()
        if not supabase:
            return False, "Failed to initialize database client. Check server logs."
        
        # Check if user already exists
        if user_exists(email):
            return False, "An account with this email already exists. Please login."
        
        # Hash password
        password_hash = hash_password(password)
        
        # Insert user into Supabase
        # Note: language field temporarily removed until database schema is updated
        data = {
            'email': email.lower(),
            'password_hash': password_hash,
            'name': name.strip(),
            'created_at': datetime.now().isoformat()
        }
        
        supabase.table('users').insert(data).execute()
        
        return True, "Account created successfully!"
    
    except Exception as e:
        print(f"Error creating user: {e}")
        return False, f"Error creating account: {str(e)}"


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[User], str]:
    """
    Authenticate user credentials against Supabase.
    
    Args:
        email: User email
        password: Plain text password
        
    Returns:
        (success: bool, user: Optional[User], message: str)
    """
    # Validate inputs
    if not email or not password:
        return False, None, "Please enter both email and password"
    
    # Check credentials first
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        return False, None, "Missing Supabase configuration. Check .env file."

    try:
        supabase = get_supabase_client()
        if not supabase:
            return False, None, "Failed to initialize database client. Check server logs."
        
        # Get user from Supabase
        response = supabase.table('users').select('*').eq('email', email.lower()).execute()
        
        if not response.data:
            return False, None, "Invalid email or password"
        
        user_data = response.data[0]
        
        # Verify password
        if not verify_password(password, user_data['password_hash']):
            return False, None, "Invalid email or password"
        
        # Update last_login timestamp
        supabase.table('users').update({
            'last_login': datetime.now().isoformat()
        }).eq('email', email.lower()).execute()
        
        # Create User object
        user = User(
            email=user_data['email'],
            password_hash=user_data['password_hash'],
            name=user_data['name'],
            language=user_data.get('language', 'en'),
            created_at=user_data['created_at'],
            last_login=datetime.now().isoformat()
        )
        
        return True, user, "Login successful!"
    
    except Exception as e:
        print(f"Authentication error: {e}")
        return False, None, f"Authentication error: {str(e)}"


def update_user_login(email: str) -> bool:
    """
    Update user's last login timestamp in Supabase.
    
    Args:
        email: User email
        
    Returns:
        True if successful
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        supabase.table('users').update({
            'last_login': datetime.now().isoformat()
        }).eq('email', email.lower()).execute()
        
        return True
    except Exception as e:
        print(f"Error updating login: {e}")
        return False


def update_user_language(email: str, language: str) -> bool:
    """
    Update user's language preference in Supabase.
    
    Args:
        email: User email
        language: Language code ('en', 'hi', 'te')
        
    Returns:
        True if successful
    """
    # Validate language
    if language not in ['en', 'hi', 'te']:
        return False
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        supabase.table('users').update({
            'language': language
        }).eq('email', email.lower()).execute()
        
        return True
    except Exception as e:
        print(f"Error updating language: {e}")
        return False


# ============================================================================
# GOOGLE OAUTH
# ============================================================================

def get_google_auth_url(redirect_to: str = None, supabase_client: Optional[Client] = None) -> Optional[str]:
    """
    Get Google OAuth URL for sign-in.
    
    Args:
        redirect_to: URL to redirect after successful login. Defaults to APP_URL env var or localhost:8501.
        supabase_client: Optional client to preserve PKCE session
        
    Returns:
        OAuth URL string or None if error
    """
    if redirect_to is None:
        redirect_to = os.environ.get("APP_URL", "http://localhost:8501")

    try:
        supabase = supabase_client if supabase_client else get_supabase_client()
        if not supabase:
            return None
        
        # Use provider='google'
        data = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": redirect_to
            }
        })
        
        return data.url
    except Exception as e:
        print(f"Error getting Google auth URL: {e}")
        return None


def exchange_code_for_session(auth_code: str, supabase_client: Optional[Client] = None) -> Tuple[bool, Optional[User], str]:
    """
    Exchange authorization code for session.
    
    Args:
        auth_code: Authorization code from URL query param
        supabase_client: Optional client to preserve PKCE session
        
    Returns:
        (success: bool, user: Optional[User], message: str)
    """
    try:
        supabase = supabase_client if supabase_client else get_supabase_client()
        if not supabase:
            return False, None, "Database connection error"
            
        # Exchange code for session
        response = supabase.auth.exchange_code_for_session({"auth_code": auth_code})
        
        if not response.user:
            return False, None, "Failed to retrieve user data"
            
        user_data = response.user
        
        # Check if user exists in our 'users' table or insert them if new
        # Note: Supabase Auth manages users separately from our public.users table 
        # unless we use triggers. To keep it simple, we'll upsert into our users table.
        
        # Extract user info
        email = user_data.email
        name = user_data.user_metadata.get('full_name', email.split('@')[0])
        
        # Upsert into public.users table to ensure we have a record
        # We might not have a password hash for OAuth users, so we can use a placeholder or handle it
        # For now, we'll try to find existing or create new
        
        if not user_exists(email):
             # Create a record in our custom users table
             # Use a dummy hash for oauth users or modify schema to allow null
             # We use a randomized password so no one can login with password
             dummy_hash = hash_password("OAUTH_USER_" + os.urandom(8).hex())
             
             data = {
                'email': email.lower(),
                'password_hash': dummy_hash,
                'name': name,
                'created_at': datetime.now().isoformat(),
                # 'language': 'en', # REMOVED: Column does not exist in schema
                'last_login': datetime.now().isoformat()
            }
             supabase.table('users').insert(data).execute()
        else:
            # Update last login
            supabase.table('users').update({
                'last_login': datetime.now().isoformat()
            }).eq('email', email.lower()).execute()
            
        # Get full user data from our table to return consistent User object
        # If get_user fails (e.g. slight delay), construct one from OAuth data
        db_user = get_user(email)
        if db_user:
            return True, db_user, "Login successful!"
            
        return True, User(
            email=email,
            password_hash="",
            name=name,
            language="en",
            created_at=datetime.now().isoformat(),
            last_login=datetime.now().isoformat()
        ), "Login successful!"
        
    except Exception as e:
        print(f"Error exchanging code: {e}")
        return False, None, f"Authentication error: {str(e)}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_user_count() -> int:
    """Get total number of registered users from Supabase"""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return 0
        
        response = supabase.table('users').select('email', count='exact').execute()
        return response.count if hasattr(response, 'count') else len(response.data)
    except Exception as e:
        print(f"Error getting user count: {e}")
        return 0


def delete_user(email: str) -> Tuple[bool, str]:
    """
    Delete user account from Supabase (admin function).
    
    Args:
        email: User email to delete
        
    Returns:
        (success: bool, message: str)
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False, "Database connection error"
        
        supabase.table('users').delete().eq('email', email.lower()).execute()
        return True, "User deleted successfully"
    except Exception as e:
        print(f"Error deleting user: {e}")
        return False, f"Error deleting user: {str(e)}"
