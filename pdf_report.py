"""
PDF Report Generator
====================

Generates professional PDF health reports from session data.
Uses ReportLab for layout and styling.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO
import base64
from datetime import datetime
from typing import Optional

from session_storage import SessionData
from translations import get_text


def generate_health_report(session: SessionData, lang: str = "en") -> bytes:
    """
    Generate a complete PDF health report from session data.
    
    Args:
        session: SessionData object containing all health information
        
    Returns:
        PDF file as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=6
    )
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    elements.append(Paragraph("Wellio Health Report", title_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Parse timestamp
    try:
        timestamp_dt = datetime.fromisoformat(session.timestamp)
        formatted_date = timestamp_dt.strftime("%d %B %Y")
        formatted_time = timestamp_dt.strftime("%I:%M %p")
    except:
        formatted_date = "N/A"
        formatted_time = "N/A"
    
    # Metadata table
    metadata = [
        ["Report Date:", formatted_date],
        ["Report Time:", formatted_time],
        ["Analysis Type:", session.analysis_type],
        ["Report ID:", session.session_id[:8]]
    ]
    
    metadata_table = Table(metadata, colWidths=[1.5*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6b7280')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1f2937')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    
    elements.append(metadata_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # ========================================================================
    # USER PROFILE
    # ========================================================================
    
    elements.append(Paragraph("User Profile", heading_style))
    
    profile_data = [
        ["Age:", f"{session.age} years", "Gender:", session.gender],
        ["Height:", f"{session.height:.1f} cm", "Weight:", f"{session.weight:.1f} kg"],
        ["BMI:", f"{session.bmi:.1f}", "Diet:", session.diet],
        ["Exercise:", session.exercise, "Sleep:", f"{session.sleep:.1f} hours/night"],
        ["Smoking:", session.smoking, "Drinking:", session.drinking]
    ]
    
    profile_table = Table(profile_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    profile_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTNAME', (3, 0), (3, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2937')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f9fafb')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f9fafb')),
    ]))
    
    elements.append(profile_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # ========================================================================
    # VITAL SIGNS
    # ========================================================================
    
    elements.append(Paragraph("Vital Signs", heading_style))
    
    vitals_data = [
        ["Metric", "Value", "Status"],
        ["Heart Rate (rPPG)", f"{session.heart_rate:.1f} BPM", session.heart_rate_confidence],
        ["Stress Index", f"{session.stress_level:.1f}/10", get_stress_label(session.stress_level)],
        ["Blood Pressure", 
         f"{session.bp_systolic:.0f}/{session.bp_diastolic:.0f} mmHg" if session.bp_systolic else "N/A",
         get_bp_label(session.bp_systolic) if session.bp_systolic else "N/A"],
        ["SpO₂", f"{session.spo2:.1f}%" if session.spo2 else "N/A",
         get_spo2_label(session.spo2) if session.spo2 else "N/A"],
        ["HRV (SDNN)", f"{session.hrv_sdnn:.1f} ms", f"{session.rr_intervals_count} beats analyzed"]
    ]
    
    vitals_table = Table(vitals_data, colWidths=[2*inch, 2*inch, 2.5*inch])
    vitals_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(vitals_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # ========================================================================
    # RISK ASSESSMENT
    # ========================================================================
    
    elements.append(Paragraph("Risk Assessment", heading_style))
    
    # Risk score with color
    risk_color = get_risk_color(session.risk_score)
    risk_data = [
        ["Risk Score:", f"{session.risk_score}/10"],
        ["Risk Level:", session.risk_level]
    ]
    
    risk_table = Table(risk_data, colWidths=[1.5*inch, 4*inch])
    risk_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f2937')),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#1f2937')),
        ('TEXTCOLOR', (1, 1), (1, 1), risk_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(risk_table)
    elements.append(Spacer(1, 0.15*inch))
    
    # Risk factors
    if session.risk_factors:
        elements.append(Paragraph("Risk Factors:", subheading_style))
        for factor in session.risk_factors:
            elements.append(Paragraph(f"• {factor}", normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    # Protective factors
    if session.protective_factors:
        elements.append(Paragraph("Protective Factors:", subheading_style))
        for factor in session.protective_factors:
            elements.append(Paragraph(f"• {factor}", normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Spacer(1, 0.1*inch))
    
    # ========================================================================
    # AI HEALTH INSIGHTS
    # ========================================================================
    
    elements.append(Paragraph("AI Health Insights", heading_style))
    
    if session.detailed_analysis:
        elements.append(Paragraph("Analysis:", subheading_style))
        elements.append(Paragraph(session.detailed_analysis, normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    if session.recommendations:
        elements.append(Paragraph("Personalized Recommendations:", subheading_style))
        for rec in session.recommendations:
            elements.append(Paragraph(f"• {rec}", normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    if session.symptoms_to_watch:
        elements.append(Paragraph("Symptoms to Watch:", subheading_style))
        for symptom in session.symptoms_to_watch:
            elements.append(Paragraph(f"• {symptom}", normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    # ========================================================================
    # SIGNAL VISUALIZATIONS (Removed per user request)
    # ========================================================================
    
    # if session.signal_plot or session.hrv_plot:
    #     elements.append(PageBreak())
    #     elements.append(Paragraph("Signal Analysis", heading_style))
    #     
    #     if session.signal_plot:
    #         try:
    #             img_data = base64.b64decode(session.signal_plot)
    #             img_buffer = BytesIO(img_data)
    #             img = RLImage(img_buffer, width=6*inch, height=4*inch)
    #             elements.append(img)
    #             elements.append(Spacer(1, 0.2*inch))
    #         except Exception as e:
    #             print(f"Error embedding signal plot: {e}")
    #     
    #     if session.hrv_plot:
    #         try:
    #             img_data = base64.b64decode(session.hrv_plot)
    #             img_buffer = BytesIO(img_data)
    #             img = RLImage(img_buffer, width=6*inch, height=3*inch)
    #             elements.append(img)
    #             elements.append(Spacer(1, 0.2*inch))
    #         except Exception as e:
    #             print(f"Error embedding HRV plot: {e}")
    
    # ========================================================================
    # DISCLAIMER
    # ========================================================================
    
    elements.append(Spacer(1, 0.3*inch))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#6b7280'),
        alignment=TA_CENTER,
        borderWidth=1,
        borderColor=colors.HexColor('#d1d5db'),
        borderPadding=10,
        backColor=colors.HexColor('#f9fafb')
    )
    
    disclaimer_text = """
    <b>IMPORTANT DISCLAIMER:</b><br/>
    This report is generated by an AI-based non-contact remote photoplethysmography (rPPG) system 
    and is intended for informational and wellness purposes only. <b>This is NOT a medical diagnosis.</b><br/>
    The measurements are experimental and may have significant error margins. Do not use this report 
    to make medical decisions. Always consult a qualified healthcare professional for medical advice, 
    diagnosis, or treatment. If you experience concerning symptoms, seek immediate medical attention.
    """
    
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stress_label(stress: float, lang: str = "en") -> str:
    """Get stress level label from numeric value"""
    if stress <= 2:
        return get_text("stress_very_low", lang)
    elif stress <= 4:
        return get_text("stress_low", lang)
    elif stress <= 6:
        return get_text("stress_moderate", lang)
    elif stress <= 8:
        return get_text("stress_high", lang)
    else:
        return get_text("stress_very_high", lang)


def get_bp_label(systolic: Optional[float], lang: str = "en") -> str:
    """Get blood pressure label from systolic value"""
    if systolic is None:
        return get_text("na", lang)
    if systolic < 90:
        return get_text("bp_low", lang)
    elif systolic <= 119:
        return get_text("bp_normal", lang)
    elif systolic <= 129:
        return get_text("bp_elevated", lang)
    elif systolic <= 139:
        return get_text("bp_stage1", lang)
    else:
        return get_text("bp_stage2", lang)


def get_spo2_label(spo2: Optional[float], lang: str = "en") -> str:
    """Get SpO2 label from numeric value"""
    if spo2 is None:
        return get_text("na", lang)
    if spo2 >= 95:
        return get_text("spo2_normal", lang)
    elif spo2 >= 92:
        return get_text("spo2_slightly_low", lang)
    elif spo2 >= 88:
        return get_text("spo2_low", lang)
    else:
        return get_text("spo2_very_low", lang)


def get_risk_color(risk_score: int) -> colors.Color:
    """Get color for risk level"""
    if risk_score <= 3:
        return colors.HexColor('#16a34a')  # Green
    elif risk_score <= 6:
        return colors.HexColor('#f59e0b')  # Orange
    else:
        return colors.HexColor('#dc2626')  # Red
