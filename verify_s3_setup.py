import os
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_s3():
    print("--- Verifying AWS S3 Configuration ---")
    
    # 1. Check Credentials presence
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION")
    bucket = os.getenv("AWS_S3_BUCKET")
    
    print(f"AWS_REGION: {region}")
    print(f"AWS_S3_BUCKET: {bucket}")
    
    if not access_key or not secret_key:
        print("❌ Error: AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is missing in .env")
        return
    
    print(f"AWS_ACCESS_KEY_ID: {'*' * (len(access_key)-4) + access_key[-4:] if access_key else 'MISSING'}")
    print(f"AWS_SECRET_ACCESS_KEY: {'[PRESENT]' if secret_key else 'MISSING'}")

    # 2. Test Connection
    print("\nAttempting to connect to S3...")
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Verify specific bucket access
        print("Checking bucket access...")
        s3.head_bucket(Bucket=bucket)
        print(f"✅ Connection Successful! Bucket '{bucket}' is accessible.")
        
        # 3. Test Upload
        print(f"\nAttempting test upload to '{bucket}'...")
        try:
            s3.put_object(
                Bucket=bucket,
                Key="test_upload.txt",
                Body=b"This is a test file from Wellio S3 verification script.",
                ContentType="text/plain"
            )
            print(f"✅ Test upload successful! (File: test_upload.txt)")
        except Exception as e:
            print(f"❌ Upload Failed: {e}")

    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    verify_s3()
