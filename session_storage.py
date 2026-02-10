"""
Session Storage Module
======================

Handles persistence of health analysis sessions for usage history.
Stores session data in JSON format with user-specific directories.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import uuid


@dataclass
class SessionData:
    """Structured storage for a single health analysis session"""
    
    # Session metadata
    session_id: str
    timestamp: str  # ISO 8601 format
    analysis_type: str  # e.g., "Health Scan"
    
    # User Profile
    age: int
    gender: str
    height: float
    weight: float
    bmi: float
    diet: str
    exercise: str
    sleep: float
    smoking: str
    drinking: str
    
    # Vitals
    heart_rate: float
    heart_rate_confidence: str
    stress_level: float
    stress_label: str
    stress_score: Optional[int]
    stress_reason: str
    bp_systolic: Optional[float]
    bp_diastolic: Optional[float]
    bp_label: str = "Unavailable"
    bp_reason: str = ""
    bp_confidence: int = 0
    bp_disclaimer: str = ""
    spo2: Optional[float] = None
    hrv_sdnn: float = 0.0
    hrv_pnn50: float = 0.0
    rr_intervals_count: int = 0
    
    # Risk Assessment
    risk_score: int = 0
    risk_level: str = "LOW"
    risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    
    # AI Insights
    detailed_analysis: str = ""
    recommendations: List[str] = field(default_factory=list)
    symptoms_to_watch: List[str] = field(default_factory=list)
    
    # Visualizations (base64 encoded PNG)
    signal_plot: Optional[str] = None
    hrv_plot: Optional[str] = None


def get_storage_base_path() -> Path:
    """Get the base storage directory for all sessions"""
    home = Path.home()
    storage_path = home / ".wellio" / "sessions"
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path


def get_user_storage_path(username: str) -> Path:
    """Get user-specific storage directory"""
    user_path = get_storage_base_path() / username
    user_path.mkdir(parents=True, exist_ok=True)
    return user_path


def save_session(username: str, session: SessionData) -> bool:
    """
    Save a session to JSON file.
    
    Args:
        username: User identifier
        session: SessionData object to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        user_path = get_user_storage_path(username)
        filename = f"{session.session_id}.json"
        filepath = user_path / filename
        
        # Convert to dict and save
        session_dict = asdict(session)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        print(f"Error saving session: {e}")
        return False


def load_session(username: str, session_id: str) -> Optional[SessionData]:
    """
    Load a specific session by ID.
    
    Args:
        username: User identifier
        session_id: Session UUID
        
    Returns:
        SessionData object or None if not found
    """
    try:
        user_path = get_user_storage_path(username)
        filepath = user_path / f"{session_id}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            session_dict = json.load(f)
        
        return SessionData(**session_dict)
    
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
        return None


def list_sessions(username: str, limit: Optional[int] = None) -> List[SessionData]:
    """
    List all sessions for a user, sorted by timestamp (newest first).
    
    Args:
        username: User identifier
        limit: Optional maximum number of sessions to return
        
    Returns:
        List of SessionData objects
    """
    try:
        user_path = get_user_storage_path(username)
        sessions = []
        
        # Find all JSON files
        for filepath in user_path.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    session_dict = json.load(f)
                sessions.append(SessionData(**session_dict))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda s: s.timestamp, reverse=True)
        
        # Apply limit if specified
        if limit:
            sessions = sessions[:limit]
        
        return sessions
    
    except Exception as e:
        print(f"Error listing sessions: {e}")
        return []


def delete_session(username: str, session_id: str) -> bool:
    """
    Delete a specific session.
    
    Args:
        username: User identifier
        session_id: Session UUID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        user_path = get_user_storage_path(username)
        filepath = user_path / f"{session_id}.json"
        
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        return False


def get_session_count(username: str) -> int:
    """Get total number of sessions for a user"""
    try:
        user_path = get_user_storage_path(username)
        return len(list(user_path.glob("*.json")))
    except Exception:
        return 0
