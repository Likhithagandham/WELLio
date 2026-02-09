"""
Health Insights Module
Generates AI-powered health explanations using the Groq API
"""

from groq import Groq
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Export GROQ_API_KEY for use in other modules
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@dataclass
class HealthInsightResponse:
    """Structured response from Groq for health insights"""
    detailed_analysis: str
    risk_factors: list[str]
    positive_indicators: list[str]
    recommendations: list[str]
    symptoms_to_watch: list[str]
    error: Optional[str] = None


def build_health_insights_prompt(
    pulse_bpm: float,
    stress_index: float,
    estimated_sbp: float,
    estimated_dbp: float,
    estimated_spo2: float,
    age: int,
    gender: str,
    height: float,
    weight: float,
    diet: str,
    exercise_frequency: str,
    sleep_hours: float,
    smoking_habits: str,
    lang: str = "en"
) -> str:
    """
    Build a structured prompt for Groq that includes vitals and profile data.
    Returns a prompt that requests 5 specific sections.
    """
    
    # Determine language for response
    lang_instruction = ""
    if lang == "hi":
        lang_instruction = "\n\nIMPORTANT: Provide your response in Hindi (हिंदी). Use Devanagari script."
    elif lang == "te":
        lang_instruction = "\n\nIMPORTANT: Provide your response in Telugu (తెలుగు). Use Telugu script."
    else:
        lang_instruction = "\n\nIMPORTANT: Provide your response in English."
    
    prompt = f"""
You are a health insights advisor. Based on the following vital signs and user profile, provide structured health insights.

VITAL SIGNS:
- Pulse: {pulse_bpm:.1f} BPM
- Stress Index: {stress_index:.1f}/10
- Estimated Systolic BP: {estimated_sbp:.0f} mmHg
- Estimated Diastolic BP: {estimated_dbp:.0f} mmHg
- Estimated SpO₂: {estimated_spo2:.1f}%

USER PROFILE:
- Age: {age} years
- Gender: {gender}
- Height: {height:.1f} cm
- Weight: {weight:.1f} kg
- Diet: {diet}
- Exercise Frequency: {exercise_frequency}
- Sleep Hours: {sleep_hours:.1f} hours/night
- Smoking Habits: {smoking_habits}

TASK: Provide health insights in the following EXACT format. Each section should be concise and informative.

1. DETAILED HEALTH ANALYSIS:
   Provide 3-4 sentences in plain language explaining what these vitals indicate about overall health. Use reassuring, informative tone. Avoid medical diagnosis.

2. RISK FACTORS:
   List 3-6 lifestyle or vital-based factors that may increase health risk, as bullet points (- format). Only mention factors supported by the data.

3. POSITIVE INDICATORS:
   List 2-4 positive health signs or healthy behaviors (- format). Highlight what's working well.

4. PERSONALIZED RECOMMENDATIONS:
   List 4-6 actionable lifestyle suggestions tailored to sleep, exercise, diet, and smoking habits (- format). Avoid medical or medication advice.

5. SYMPTOMS TO WATCH:
   List 3-5 general warning symptoms to be aware of (- format). Use conditional language like "may", "could". Do not claim diagnosis.

IMPORTANT: Start each section with its number and title (e.g., "1. DETAILED HEALTH ANALYSIS:"), then provide the content.
Keep language simple and suitable for non-medical users.{lang_instruction}
"""
    
    return prompt


def parse_groq_response(response_text: str) -> HealthInsightResponse:
    """
    Parse the structured Groq response into the 5 sections.
    Returns a HealthInsightResponse dataclass.
    """
    
    sections = {
        "detailed_analysis": "",
        "risk_factors": [],
        "positive_indicators": [],
        "recommendations": [],
        "symptoms_to_watch": [],
    }
    
    # Split by numbered sections
    lines = response_text.strip().split("\n")
    current_section = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        
        # Detect section headers
        if line.startswith("1. DETAILED HEALTH ANALYSIS"):
            if current_section and current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = "detailed_analysis"
            current_content = []
        elif line.startswith("2. RISK FACTORS"):
            if current_section and current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = "risk_factors"
            current_content = []
        elif line.startswith("3. POSITIVE INDICATORS"):
            if current_section and current_content:
                if current_section in ["risk_factors", "positive_indicators", "recommendations", "symptoms_to_watch"]:
                    sections[current_section] = [item.lstrip("- ") for item in current_content if item.strip().startswith("-")]
                else:
                    sections[current_section] = "\n".join(current_content).strip()
            current_section = "positive_indicators"
            current_content = []
        elif line.startswith("4. PERSONALIZED RECOMMENDATIONS"):
            if current_section and current_content:
                if current_section in ["risk_factors", "positive_indicators", "recommendations", "symptoms_to_watch"]:
                    sections[current_section] = [item.lstrip("- ") for item in current_content if item.strip().startswith("-")]
                else:
                    sections[current_section] = "\n".join(current_content).strip()
            current_section = "recommendations"
            current_content = []
        elif line.startswith("5. SYMPTOMS TO WATCH"):
            if current_section and current_content:
                if current_section in ["risk_factors", "positive_indicators", "recommendations", "symptoms_to_watch"]:
                    sections[current_section] = [item.lstrip("- ") for item in current_content if item.strip().startswith("-")]
                else:
                    sections[current_section] = "\n".join(current_content).strip()
            current_section = "symptoms_to_watch"
            current_content = []
        elif line and current_section:
            current_content.append(line)
    
    # Capture the last section
    if current_section and current_content:
        if current_section in ["risk_factors", "positive_indicators", "recommendations", "symptoms_to_watch"]:
            sections[current_section] = [item.lstrip("- ") for item in current_content if item.strip().startswith("-")]
        else:
            sections[current_section] = "\n".join(current_content).strip()
    
    return HealthInsightResponse(
        detailed_analysis=sections["detailed_analysis"],
        risk_factors=sections["risk_factors"],
        positive_indicators=sections["positive_indicators"],
        recommendations=sections["recommendations"],
        symptoms_to_watch=sections["symptoms_to_watch"],
    )


def get_health_insights(
    pulse_bpm: float,
    stress_index: float,
    estimated_sbp: float,
    estimated_dbp: float,
    estimated_spo2: float,
    age: int,
    gender: str,
    height: float,
    weight: float,
    diet: str,
    exercise_frequency: str,
    sleep_hours: float,
    smoking_habits: str,
    api_key: str = None,
    lang: str = "en"
) -> HealthInsightResponse:
    """
    Main function to fetch and parse health insights from Groq.
    
    Args:
        vitals and profile data as above
        api_key: Groq API key (default provided)
    
    Returns:
        HealthInsightResponse with parsed insights or error message
    """
    
    try:
        # Use GROQ_API_KEY constant if no api_key provided
        if api_key is None:
            api_key = GROQ_API_KEY
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Build prompt
        prompt = build_health_insights_prompt(
            pulse_bpm,
            stress_index,
            estimated_sbp,
            estimated_dbp,
            estimated_spo2,
            age,
            gender,
            height,
            weight,
            diet,
            exercise_frequency,
            sleep_hours,
            smoking_habits,
            lang
        )
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2000,
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Parse response
        insights = parse_groq_response(response_text)
        return insights
        
    except Exception as e:
        return HealthInsightResponse(
            detailed_analysis="",
            risk_factors=[],
            positive_indicators=[],
            recommendations=[],
            symptoms_to_watch=[],
            error=f"Health insights unavailable at the moment. Error: {str(e)}"
        )
