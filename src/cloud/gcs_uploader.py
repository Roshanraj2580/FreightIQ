"""
FreightIQ - Stage 2: Google Cloud Storage (GCS) Sync
Uploads clean parquet and raw datasets to GCS buckets for enterprise data lake ingestion.
"""

import os
import sys

def upload_to_gcs(local_file: str, bucket_name: str, destination_blob_name: str, project_id: str = None) -> bool:
    """
    Uploads a local file to a Google Cloud Storage bucket.
    """
    if not os.path.exists(local_file):
        print(f"[GCS Uploader] Error: Local file not found: {local_file}")
        return False
        
    try:
        from google.cloud import storage
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError:
        print("[GCS Uploader] google-cloud-storage not installed. Install via pip.")
        return False

    try:
        client = storage.Client(project=project_id) if project_id else storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Check if bucket exists, create if not
        if not bucket.exists():
            print(f"[GCS Uploader] Bucket {bucket_name} does not exist. Creating in us-central1...")
            bucket = client.create_bucket(bucket_name, location="us-central1")
            
        blob = bucket.blob(destination_blob_name)
        print(f"[GCS Uploader] Uploading {local_file} -> gs://{bucket_name}/{destination_blob_name}...")
        blob.upload_from_filename(local_file)
        print(f"[GCS Uploader] Upload complete: gs://{bucket_name}/{destination_blob_name}")
        return True
    except DefaultCredentialsError:
        print("[GCS Uploader] No GCP credentials detected. In local mode, files remain stored in data/processed/.")
        print("[GCS Uploader] To enable cloud sync, set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'.")
        return False
    except Exception as e:
        print(f"[GCS Uploader] Cloud sync encountered an exception: {e}")
        return False

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    bucket = os.getenv("GCS_BUCKET_NAME", "freightiq-raw-and-curated")
    project = os.getenv("GCP_PROJECT_ID", None)
    curated_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "clean_freight_features.parquet")
    
    upload_to_gcs(curated_file, bucket, "curated/clean_freight_features.parquet", project)
