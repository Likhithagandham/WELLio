# AWS S3 Reports Integration Guide

This guide explains how to securely upload **PDF Health Reports** to AWS S3 using Supabase Authentication.

> [!IMPORTANT]
> **Video files are NOT stored in S3.** Only the generated PDF reports are saved to your cloud storage for user history.

## Architecture

1.  **Frontend (Streamlit)**: Generates a PDF report after analysis.
2.  **Frontend**: Checks user authentication.
3.  **Frontend**: Uploads the PDF directly to S3 using server-side AWS credentials.
4.  **Frontend**: Saves file metadata to `user_files` table in Supabase.
5.  **Backend (FastAPI)**: Provides an `/api/upload/url` endpoint (restricted to PDF) if needed for external clients.

## Prerequisites

1.  **AWS S3 Bucket**: Create a bucket (e.g., `wellio-uploads`) in `ap-south-1`.
2.  **Environment Variables**:
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_REGION` (ap-south-1)
    - `AWS_S3_BUCKET` (your bucket name)

## How to Test

1.  **Start the App**: `streamlit run rppg_streamlit_ui.py`
2.  **Run an Analysis**: Upload a video or use the live camera.
3.  **Generate Report**: Click "Generate PDF Report" in the history/results section.
4.  **Automatic Save**: The report is silently uploaded in the background.
5.  **Verify**:
    - Check your S3 bucket folder `reports/<user_email>/`.
    - Check the `user_files` table in Supabase.
