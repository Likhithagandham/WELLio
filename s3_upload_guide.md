# AWS S3 Upload Integration Guide

This guide explains how to securely upload files to AWS S3 using Supabase Authentication and a Python FastAPI backend.

## Architecture

1.  **Frontend**: Checks user authentication using Supabase Auth.
2.  **Frontend**: Requests a **Presigned Upload URL** from the Backend, passing the Supabase JWT.
3.  **Backend**: Verifies the JWT with Supabase.
4.  **Backend**: Generates a secure, temporary S3 Presigned URL using AWS Credentials (server-side only).
5.  **Backend**: Returns the URL to the Frontend.
6.  **Frontend**: Uploads the file **directly** to S3 using the Presigned URL.
7.  **Frontend**: Saves file metadata (filename, S3 key, etc.) to the `user_files` table in Supabase.

## Prerequisites

1.  **AWS S3 Bucket**: Create a bucket (e.g., `wellio-uploads`) in `ap-south-1`.
2.  **CORS Configuration**: Allow PUT requests from your frontend origin (e.g., `http://localhost:3000` or `http://localhost:8501`).
    ```json
    [
        {
            "AllowedHeaders": ["*"],
            "AllowedMethods": ["PUT", "POST", "GET"],
            "AllowedOrigins": ["*"],
            "ExposeHeaders": []
        }
    ]
    ```
3.  **Environment Variables**:
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_REGION` (ap-south-1)
    - `AWS_S3_BUCKET` (your bucket name)
    - `SUPABASE_URL`
    - `SUPABASE_KEY`

## Components

### 1. Database Schema (`supabase_storage_schema.sql`)
Creates the `user_files` table with RLS policies to ensure users can only access their own files.

### 2. Backend (`rppg_fastapi.py` & `s3_utils.py`)
- **`s3_utils.py`**: Helper to generate presigned URLs using `boto3`.
- **`rppg_fastapi.py`**: Exposes `POST /api/upload/url`. Verifies Supabase Bearer token and returns the URL.

### 3. Frontend (`frontend_upload_example.js`)
JavaScript function `uploadFileToS3(file)` that orchestrates the flow.

## How to Test

1.  Ensure your `.env` has the AWS credentials.
2.  Run the backend: `uvicorn rppg_fastapi:app --reload`.
3.  Use the `frontend_upload_example.js` logic in your frontend application.
4.  Check the `user_files` table in Supabase to verify the metadata is saved.
5.  Check your S3 bucket to verify the file is uploaded.
