"""
Patient Health Assistant Chatbot
=================================

AI-powered health assistant that helps users understand their health data,
trends, and provides general wellness guidance with strict safety guardrails.

IMPORTANT: This chatbot is NOT a medical professional and does NOT:
- Diagnose diseases
- Prescribe medications
- Replace professional medical care
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict
import json
from pathlib import Path

try:
    from groq import Groq
    HAVE_GROQ = True
except ImportError:
    HAVE_GROQ = False

from session_storage import SessionData, list_sessions
from trend_analysis import get_trend_analysis, TrendAnalysis
from translations import get_text


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChatMessage:
    """Single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str  # ISO format
    risk_level: str = "low"  # "low", "medium", "high"
    audio_bytes: Optional[bytes] = None  # TTS audio data


@dataclass
class ChatContext:
    """User context for chatbot"""
    username: str
    user_profile: Dict
    latest_session: Optional[SessionData]
    recent_sessions: List[SessionData]
    trend_analysis: Optional[TrendAnalysis]
    chat_history: List[ChatMessage]


# ============================================================================
# SAFETY CONFIGURATION
# ============================================================================

HIGH_RISK_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "cannot breathe",
    "breathless", "shortness of breath", "fainting", "fainted", "unconscious",
    "severe pain", "bleeding", "vomiting blood", "seizure", "convulsion",
    "extreme dizziness", "confusion", "slurred speech", "paralysis",
    "numbness", "weakness", "blurred vision", "double vision",
    "severe headache", "migraine", "passing out", "collapsed"
]



SYSTEM_PROMPT = """You are a Patient Health Assistant for the Wellio health monitoring application.

YOUR ROLE:
- Help users understand their health data and trends
- Provide general wellness and lifestyle advice
- Explain vitals and health metrics in simple terms
- Encourage professional care when needed
- Be empathetic, supportive, and informative

STRICT RULES YOU MUST FOLLOW:
1. NEVER diagnose diseases or medical conditions
2. NEVER prescribe medications or treatments
3. NEVER claim to replace a doctor or medical professional
4. ALWAYS use suggestive, non-authoritative language
5. ALWAYS escalate severe or worsening symptoms to professional care
6. ONLY reference data that is provided in the user context
7. Do NOT make up or infer data that isn't provided

ALLOWED LANGUAGE PATTERNS:
✓ "You may consider..."
✓ "In general..."
✓ "Commonly associated with..."
✓ "Many people find that..."
✓ "Based on your data..."
✓ "Your readings show..."

FORBIDDEN LANGUAGE PATTERNS:
✗ "You have [disease]..."
✗ "You should take [medication]..."
✗ "This means you are suffering from..."
✗ "I diagnose you with..."
✗ "You need [specific treatment]..."

RESPONSE STYLE:
- KEEP IT SIMPLE AND STRAIGHTFORWARD.
- MAX 3-4 SENTENCES TOTAL.
- No long paragraphs.
- Direct answers only.
- If you don't have enough data, say so in 1 sentence.

WHEN TO ESCALATE:
If the user mentions:
- Severe symptoms (chest pain, breathlessness, fainting)
- Worsening trends in critical vitals
- Symptoms that could indicate emergency
→ Recommend immediate professional medical attention

Remember: You are a helpful assistant, not a doctor. Your goal is to help users understand their data and make informed decisions about seeking professional care.
"""


# ============================================================================
# RISK DETECTION
# ============================================================================

def analyze_risk_level(user_message: str, context: ChatContext) -> str:
    """
    Analyze risk level based on message content and user data.
    
    Args:
        user_message: User's message
        context: User context with health data
        
    Returns:
        "low", "medium", or "high"
    """
    message_lower = user_message.lower()
    
    # Check for high-risk keywords
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in message_lower:
            return "high"
    
    # Check latest vitals for concerning values
    if context.latest_session:
        session = context.latest_session
        
        # Critical SpO2
        if session.spo2 and session.spo2 < 90:
            return "high"
        
        # Extreme heart rate
        if session.heart_rate:
            if session.heart_rate > 120 or session.heart_rate < 40:
                return "high"
        
        # Very high stress
        if session.stress_level and session.stress_level > 8:
            return "medium"
    
    # Check trends for concerning patterns
    if context.trend_analysis:
        concerning_count = 0
        
        if context.trend_analysis.heart_rate:
            if context.trend_analysis.heart_rate.trend_classification == "Concerning":
                concerning_count += 1
        
        if context.trend_analysis.spo2:
            if context.trend_analysis.spo2.trend_classification == "Concerning":
                concerning_count += 1
        
        if context.trend_analysis.stress_level:
            if context.trend_analysis.stress_level.trend_classification == "Worsening":
                concerning_count += 1
        
        if concerning_count >= 2:
            return "high"
        elif concerning_count == 1:
            return "medium"
    
    return "low"


def should_escalate(risk_level: str) -> bool:
    """Determine if medical escalation is needed"""
    return risk_level == "high"


# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def build_chatbot_context(
    username: str,
    user_profile: Dict,
    latest_session: Optional[SessionData] = None,
    chat_history: Optional[List[ChatMessage]] = None
) -> ChatContext:
    """
    Build comprehensive context for chatbot.
    
    Args:
        username: User identifier
        user_profile: User profile data
        latest_session: Current session data (if available)
        chat_history: Previous chat messages
        
    Returns:
        ChatContext object
    """
    # Load historical sessions
    sessions = list_sessions(username)
    recent_sessions = sessions[:10] if sessions else []
    
    # Use latest from history if not provided
    if not latest_session and sessions:
        latest_session = sessions[0]
    
    # Get trend analysis
    trend_analysis = None
    if len(sessions) >= 2:
        try:
            trend_analysis = get_trend_analysis(username, days=30, user_age=user_profile.get("age", 30))
        except Exception:
            pass
    
    # Use provided chat history or empty list
    if chat_history is None:
        chat_history = []
    
    return ChatContext(
        username=username,
        user_profile=user_profile,
        latest_session=latest_session,
        recent_sessions=recent_sessions,
        trend_analysis=trend_analysis,
        chat_history=chat_history
    )


def format_context_for_prompt(context: ChatContext) -> str:
    """
    Format context into readable text for LLM prompt.
    
    Args:
        context: ChatContext object
        
    Returns:
        Formatted context string
    """
    parts = []
    
    # User profile
    parts.append("USER PROFILE:")
    parts.append(f"- Age: {context.user_profile.get('age', 'Unknown')}")
    parts.append(f"- Gender: {context.user_profile.get('gender', 'Unknown')}")
    
    lifestyle = context.user_profile.get('lifestyle', {})
    if lifestyle:
        parts.append(f"- Diet: {lifestyle.get('diet', 'Unknown')}")
        parts.append(f"- Exercise: {lifestyle.get('exercise', 'Unknown')}")
        parts.append(f"- Sleep: {lifestyle.get('sleep', 'Unknown')} hours/night")
        parts.append(f"- Smoking: {lifestyle.get('smoking', 'Unknown')}")
        parts.append(f"- Drinking: {lifestyle.get('drinking', 'Unknown')}")
    
    # Latest vitals
    if context.latest_session:
        session = context.latest_session
        try:
            timestamp = datetime.fromisoformat(session.timestamp)
            date_str = timestamp.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            date_str = "Recent"
        
        parts.append(f"\nLATEST ANALYSIS ({date_str}):")
        parts.append(f"- Heart Rate: {session.heart_rate:.1f} BPM ({session.heart_rate_confidence})")
        parts.append(f"- Stress Level: {session.stress_level:.1f}/10")
        
        if session.spo2:
            parts.append(f"- SpO₂: {session.spo2:.1f}%")
        
        if session.bp_systolic and session.bp_diastolic:
            parts.append(f"- Blood Pressure: {session.bp_systolic:.0f}/{session.bp_diastolic:.0f} mmHg")
        
        parts.append(f"- Risk Score: {session.risk_score}/10 ({session.risk_level})")
        
        if session.risk_factors:
            parts.append(f"- Risk Factors: {', '.join(session.risk_factors[:3])}")
    
    # Trends
    if context.trend_analysis:
        parts.append(f"\nHEALTH TRENDS (Last 30 days, {context.trend_analysis.session_count} analyses):")
        
        if context.trend_analysis.heart_rate:
            hr = context.trend_analysis.heart_rate
            parts.append(f"- Heart Rate: {hr.average:.1f} BPM avg, {hr.trend_direction} trend ({hr.trend_classification})")
        
        if context.trend_analysis.stress_level:
            stress = context.trend_analysis.stress_level
            parts.append(f"- Stress Level: {stress.average:.1f}/10 avg, {stress.trend_direction} trend ({stress.trend_classification})")
        
        if context.trend_analysis.bp_systolic:
            bp = context.trend_analysis.bp_systolic
            parts.append(f"- Blood Pressure: {bp.average:.0f} mmHg avg, {bp.trend_direction} trend ({bp.trend_classification})")
        
        if context.trend_analysis.spo2:
            spo2 = context.trend_analysis.spo2
            parts.append(f"- SpO₂: {spo2.average:.1f}% avg, {spo2.trend_direction} trend ({spo2.trend_classification})")
    
    # History summary
    if context.recent_sessions:
        parts.append(f"\nHISTORY: {len(context.recent_sessions)} recent analyses available")
    else:
        parts.append("\nHISTORY: No previous analyses recorded")
    
    return "\n".join(parts)


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def generate_chatbot_response(
    user_message: str,
    context: ChatContext,
    groq_api_key: str,
    lang: str = "en"
) -> ChatMessage:
    """
    Generate chatbot response using Groq API.
    
    Args:
        user_message: User's message
        context: User context
        groq_api_key: Groq API key
        
    Returns:
        ChatMessage with response
    """
    if not HAVE_GROQ:
        return ChatMessage(
            role="assistant",
            content=get_text("chatbot_unavailable", lang),
            timestamp=datetime.now().isoformat(),
            risk_level="low"
        )
    
    # Analyze risk level
    risk_level = analyze_risk_level(user_message, context)
    
    # Build prompt
    context_str = format_context_for_prompt(context)
    
    # Include recent chat history for context
    chat_history_str = ""
    if context.chat_history:
        recent_chat = context.chat_history[-4:]  # Last 4 messages
        chat_history_str = "\n\nRECENT CONVERSATION:\n"
        for msg in recent_chat:
            chat_history_str += f"{msg.role.upper()}: {msg.content}\n"
    
    # Determine language name
    from translations import LANGUAGES
    lang_name = LANGUAGES.get(lang, {}).get("name", "English")
    
    user_prompt = f"""USER CONTEXT:
{context_str}
{chat_history_str}

USER QUESTION: {user_message}

IMPORTANT INSTRUCTION: You must respond in {lang_name}.

Please provide a helpful, informative response based on the user's data and question. Remember to follow all the rules in your system prompt."""
    
    try:
        client = Groq(api_key=groq_api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Filter unsafe language (basic check)
        content = filter_unsafe_response(content)
        
        return ChatMessage(
            role="assistant",
            content=content,
            timestamp=datetime.now().isoformat(),
            risk_level=risk_level
        )
    
    except Exception as e:
        print(f"Chatbot error: {e}")
        return ChatMessage(
            role="assistant",
            content=get_text("chatbot_error", lang),
            timestamp=datetime.now().isoformat(),
            risk_level="low"
        )


def filter_unsafe_response(response: str) -> str:
    """
    Filter out potentially unsafe language from response.
    
    Args:
        response: LLM response
        
    Returns:
        Filtered response
    """
    # Check for forbidden phrases (basic filtering)
    forbidden_phrases = [
        "you have ", "you are diagnosed", "you suffer from",
        "take this medication", "i diagnose", "prescription for"
    ]
    
    response_lower = response.lower()
    for phrase in forbidden_phrases:
        if phrase in response_lower:
            # Add a safety note (using English as this is a fallback safety measure)
            response += "\n\n" + get_text("chatbot_safety_note", "en")
            break
    
    return response


# ============================================================================
# CHAT HISTORY STORAGE
# ============================================================================

def get_chat_storage_path(username: str) -> Path:
    """Get path to user's chat history file"""
    base_path = Path.home() / ".wellio" / "chats"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / f"{username}_chat.json"


def save_chat_history(username: str, messages: List[ChatMessage]) -> bool:
    """
    Save chat history for a user.
    
    Args:
        username: User identifier
        messages: List of chat messages
        
    Returns:
        True if successful
    """
    try:
        filepath = get_chat_storage_path(username)
        
        # Convert to dict
        messages_dict = [asdict(msg) for msg in messages]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages_dict, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return False


def load_chat_history(username: str) -> List[ChatMessage]:
    """
    Load chat history for a user.
    
    Args:
        username: User identifier
        
    Returns:
        List of chat messages
    """
    try:
        filepath = get_chat_storage_path(username)
        
        if not filepath.exists():
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            messages_dict = json.load(f)
        
        # Convert to ChatMessage objects
        messages = [ChatMessage(**msg) for msg in messages_dict]
        
        return messages
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []


def clear_chat_history(username: str) -> bool:
    """
    Clear chat history for a user.
    
    Args:
        username: User identifier
        
    Returns:
        True if successful
    """
    try:
        filepath = get_chat_storage_path(username)
        if filepath.exists():
            filepath.unlink()
        return True
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        return False
