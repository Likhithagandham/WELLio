"""
Trend Analysis Module
=====================

Analyzes historical health data to detect trends, patterns, and changes over time.
Provides personalized insights based on user profile and health metrics.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np
from scipy import stats

from session_storage import SessionData, list_sessions


@dataclass
class TrendMetrics:
    """Metrics for a single health parameter over time"""
    metric_name: str
    values: List[float]
    timestamps: List[datetime]
    average: float
    min_value: float
    max_value: float
    std_dev: float
    trend_slope: float  # positive = increasing, negative = decreasing
    trend_direction: str  # "up", "down", "stable"
    trend_classification: str  # "Improving", "Worsening", "Stable", "Concerning"
    percent_change: float  # change from first to last value
    session_count: int


@dataclass
class TrendAnalysis:
    """Complete trend analysis for all metrics"""
    username: str
    time_period_days: int
    start_date: datetime
    end_date: datetime
    session_count: int
    
    # Metrics
    heart_rate: Optional[TrendMetrics]
    stress_level: Optional[TrendMetrics]
    bp_systolic: Optional[TrendMetrics]
    bp_diastolic: Optional[TrendMetrics]
    spo2: Optional[TrendMetrics]
    
    # Insights
    summary: str
    key_findings: List[str]
    recommendations: List[str]


def get_sessions_in_period(username: str, days: int) -> List[SessionData]:
    """
    Get all sessions for a user within the last N days.
    
    Args:
        username: User identifier
        days: Number of days to look back
        
    Returns:
        List of SessionData objects within the time period
    """
    all_sessions = list_sessions(username)
    
    if not all_sessions:
        return []
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    filtered_sessions = []
    for session in all_sessions:
        try:
            session_date = datetime.fromisoformat(session.timestamp)
            if session_date >= cutoff_date:
                filtered_sessions.append(session)
        except Exception:
            continue
    
    # Sort by timestamp (oldest first for trend analysis)
    filtered_sessions.sort(key=lambda s: s.timestamp)
    
    return filtered_sessions


def extract_metric_data(sessions: List[SessionData], metric_name: str) -> Tuple[List[float], List[datetime]]:
    """
    Extract time-series data for a specific metric from sessions.
    
    Args:
        sessions: List of SessionData objects
        metric_name: Name of metric to extract
        
    Returns:
        Tuple of (values, timestamps)
    """
    values = []
    timestamps = []
    
    for session in sessions:
        try:
            timestamp = datetime.fromisoformat(session.timestamp)
            
            # Extract value based on metric name
            if metric_name == "heart_rate":
                value = session.heart_rate
            elif metric_name == "stress_level":
                value = session.stress_level
            elif metric_name == "bp_systolic":
                value = session.bp_systolic
            elif metric_name == "bp_diastolic":
                value = session.bp_diastolic
            elif metric_name == "spo2":
                value = session.spo2
            else:
                continue
            
            # Only include non-None values
            if value is not None and not np.isnan(value):
                values.append(float(value))
                timestamps.append(timestamp)
        
        except Exception:
            continue
    
    return values, timestamps


def calculate_trend(values: List[float], timestamps: List[datetime]) -> Tuple[float, str]:
    """
    Calculate trend slope and direction using linear regression.
    
    Args:
        values: List of metric values
        timestamps: List of corresponding timestamps
        
    Returns:
        Tuple of (slope, direction)
    """
    if len(values) < 2:
        return 0.0, "stable"
    
    # Convert timestamps to numeric (days since first measurement)
    first_time = timestamps[0]
    x = np.array([(t - first_time).total_seconds() / 86400 for t in timestamps])
    y = np.array(values)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Determine direction based on slope and significance
    if abs(slope) < 0.01:  # Very small slope
        direction = "stable"
    elif slope > 0:
        direction = "up"
    else:
        direction = "down"
    
    return float(slope), direction


def classify_trend(
    metric_name: str,
    slope: float,
    direction: str,
    average: float,
    user_age: int = 30
) -> str:
    """
    Classify trend as Improving, Worsening, Stable, or Concerning.
    
    Args:
        metric_name: Name of the metric
        slope: Trend slope
        direction: Trend direction
        average: Average value
        user_age: User's age for context
        
    Returns:
        Classification string
    """
    
    if metric_name == "heart_rate":
        # Lower HR generally better (if not too low)
        if direction == "stable":
            return "Stable"
        elif direction == "down":
            if average > 100:
                return "Improving"  # High HR coming down
            elif average < 50:
                return "Concerning"  # Too low
            else:
                return "Stable"
        else:  # up
            if average < 60:
                return "Stable"  # Low HR increasing to normal
            elif average > 100:
                return "Worsening"  # High HR getting higher
            else:
                return "Stable"
    
    elif metric_name == "stress_level":
        # Lower stress is better
        if direction == "stable":
            return "Stable"
        elif direction == "down":
            return "Improving"
        else:  # up
            if average > 7:
                return "Concerning"
            else:
                return "Worsening"
    
    elif metric_name == "bp_systolic":
        # Target: 120 mmHg
        if direction == "stable":
            return "Stable"
        elif average < 120:
            # Below target
            if direction == "up":
                return "Improving"  # Moving towards 120
            else:
                return "Stable"
        else:
            # Above target
            if direction == "down":
                return "Improving"  # Coming down to 120
            else:
                return "Worsening"  # Getting higher
    
    elif metric_name == "bp_diastolic":
        # Target: 80 mmHg
        if direction == "stable":
            return "Stable"
        elif average < 80:
            if direction == "up":
                return "Improving"
            else:
                return "Stable"
        else:
            if direction == "down":
                return "Improving"
            else:
                return "Worsening"
    
    elif metric_name == "spo2":
        # Higher SpO2 is better (target: 95-100%)
        if direction == "stable":
            return "Stable"
        elif direction == "up":
            if average < 95:
                return "Improving"
            else:
                return "Stable"
        else:  # down
            if average < 92:
                return "Concerning"
            elif average < 95:
                return "Worsening"
            else:
                return "Stable"
    
    return "Stable"


def analyze_metric(
    sessions: List[SessionData],
    metric_name: str,
    display_name: str,
    user_age: int = 30
) -> Optional[TrendMetrics]:
    """
    Perform complete trend analysis for a single metric.
    
    Args:
        sessions: List of SessionData objects
        metric_name: Internal metric name
        display_name: Display name for the metric
        user_age: User's age
        
    Returns:
        TrendMetrics object or None if insufficient data
    """
    values, timestamps = extract_metric_data(sessions, metric_name)
    
    if len(values) < 2:
        return None
    
    # Calculate statistics
    average = float(np.mean(values))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    std_dev = float(np.std(values))
    
    # Calculate trend
    slope, direction = calculate_trend(values, timestamps)
    
    # Classify trend
    classification = classify_trend(metric_name, slope, direction, average, user_age)
    
    # Calculate percent change
    if values[0] != 0:
        percent_change = ((values[-1] - values[0]) / values[0]) * 100
    else:
        percent_change = 0.0
    
    return TrendMetrics(
        metric_name=display_name,
        values=values,
        timestamps=timestamps,
        average=average,
        min_value=min_value,
        max_value=max_value,
        std_dev=std_dev,
        trend_slope=slope,
        trend_direction=direction,
        trend_classification=classification,
        percent_change=percent_change,
        session_count=len(values)
    )


def generate_trend_summary(trend_analysis: TrendAnalysis) -> Tuple[str, List[str], List[str]]:
    """
    Generate human-readable summary and recommendations.
    
    Args:
        trend_analysis: TrendAnalysis object
        
    Returns:
        Tuple of (summary, key_findings, recommendations)
    """
    findings = []
    recommendations = []
    
    # Analyze each metric
    if trend_analysis.heart_rate:
        hr = trend_analysis.heart_rate
        if hr.trend_classification == "Improving":
            findings.append(f"Heart rate is improving (avg: {hr.average:.1f} BPM)")
        elif hr.trend_classification == "Worsening":
            findings.append(f"Heart rate is increasing (avg: {hr.average:.1f} BPM)")
            recommendations.append("Consider stress management techniques and regular exercise")
        elif hr.trend_classification == "Concerning":
            findings.append(f"Heart rate shows concerning pattern (avg: {hr.average:.1f} BPM)")
            recommendations.append("Consult a healthcare professional about your heart rate")
    
    if trend_analysis.stress_level:
        stress = trend_analysis.stress_level
        if stress.trend_classification == "Improving":
            findings.append(f"Stress levels are decreasing ({stress.percent_change:.1f}% improvement)")
            recommendations.append("Continue your current stress management practices")
        elif stress.trend_classification == "Worsening":
            findings.append(f"Stress levels are increasing (avg: {stress.average:.1f}/10)")
            recommendations.append("Try meditation, deep breathing, or regular exercise to manage stress")
    
    if trend_analysis.bp_systolic:
        bp = trend_analysis.bp_systolic
        if bp.trend_classification == "Improving":
            findings.append(f"Blood pressure is improving (avg: {bp.average:.0f} mmHg systolic)")
        elif bp.trend_classification == "Worsening":
            findings.append(f"Blood pressure is increasing (avg: {bp.average:.0f} mmHg systolic)")
            recommendations.append("Monitor sodium intake and maintain regular physical activity")
    
    if trend_analysis.spo2:
        spo2 = trend_analysis.spo2
        if spo2.trend_classification == "Concerning":
            findings.append(f"Oxygen saturation is concerning (avg: {spo2.average:.1f}%)")
            recommendations.append("Consult a healthcare professional about your oxygen levels")
    
    # Generate summary
    if not findings:
        summary = f"Analyzed {trend_analysis.session_count} sessions over {trend_analysis.time_period_days} days. All metrics are stable."
    else:
        summary = f"Analyzed {trend_analysis.session_count} sessions over {trend_analysis.time_period_days} days. "
        summary += " ".join(findings[:2])  # Top 2 findings
    
    if not recommendations:
        recommendations.append("Continue monitoring your health regularly")
    
    return summary, findings, recommendations


def get_trend_analysis(username: str, days: int = 30, user_age: int = 30) -> Optional[TrendAnalysis]:
    """
    Perform complete trend analysis for a user.
    
    Args:
        username: User identifier
        days: Number of days to analyze
        user_age: User's age for personalized insights
        
    Returns:
        TrendAnalysis object or None if insufficient data
    """
    sessions = get_sessions_in_period(username, days)
    
    if len(sessions) < 2:
        return None
    
    # Analyze each metric
    hr_trend = analyze_metric(sessions, "heart_rate", "Heart Rate", user_age)
    stress_trend = analyze_metric(sessions, "stress_level", "Stress Level", user_age)
    bp_sys_trend = analyze_metric(sessions, "bp_systolic", "Systolic BP", user_age)
    bp_dia_trend = analyze_metric(sessions, "bp_diastolic", "Diastolic BP", user_age)
    spo2_trend = analyze_metric(sessions, "spo2", "SpO₂", user_age)
    
    # Create trend analysis object
    trend_analysis = TrendAnalysis(
        username=username,
        time_period_days=days,
        start_date=datetime.fromisoformat(sessions[0].timestamp),
        end_date=datetime.fromisoformat(sessions[-1].timestamp),
        session_count=len(sessions),
        heart_rate=hr_trend,
        stress_level=stress_trend,
        bp_systolic=bp_sys_trend,
        bp_diastolic=bp_dia_trend,
        spo2=spo2_trend,
        summary="",
        key_findings=[],
        recommendations=[]
    )
    
    # Generate insights
    summary, findings, recommendations = generate_trend_summary(trend_analysis)
    trend_analysis.summary = summary
    trend_analysis.key_findings = findings
    trend_analysis.recommendations = recommendations
    
    return trend_analysis
