import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_s3_client():
    """Create a boto3 S3 client using environment variables."""
    try:
        return boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "ap-south-1")
        )
    except Exception as e:
        logger.error(f"Error creating S3 client: {e}")
        return None

def generate_presigned_url(
    bucket_name: str, 
    object_name: str, 
    expiration: int = 3600,
    content_type: str = "application/octet-stream"
) -> Optional[Dict[str, str]]:
    """
    Generate a presigned URL to share an S3 object.

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :param content_type: MIME type of the file
    :return: Dictionary with 'url' and 'fields' if successful, else None
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None

    try:
        # Generate a presigned URL for the S3 object
        # We use generate_presigned_url for simple PUT requests
        # Or generate_presigned_post for POST requests (which handle fields better)
        # Here we use generate_presigned_url for PUT as requested for simplicity
        
        url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_name,
                'ContentType': content_type
            },
            ExpiresIn=expiration
        )
        return {"url": url, "method": "PUT"}
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None

def generate_presigned_post(
    bucket_name: str, 
    object_name: str,
    expiration: int = 3600
) -> Optional[Dict]:
    """
    Generate a presigned POST URL (more secure, supports size limits directly).
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None

    try:
        response = s3_client.generate_presigned_post(
            Bucket=bucket_name,
            Key=object_name,
            Fields=None,
            Conditions=[
                ["content-length-range", 0, 524288000] # 500 MB limit
            ],
            ExpiresIn=expiration
        )
        return response
    except ClientError as e:
        logger.error(f"Error generating presigned POST: {e}")
        return None
