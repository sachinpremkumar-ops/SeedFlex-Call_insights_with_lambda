import boto3
import logging
import psycopg2
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
import json
from psycopg2.extras import RealDictCursor
import botocore.exceptions

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "experimentbucket1234")
PROCESSING_PREFIX = os.getenv("S3_PROCESSING_PREFIX", "processing/")
PROCESSED_PREFIX = os.getenv("S3_PROCESSED_PREFIX", "processed_latest/")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)    


s3_client = boto3.client('s3')

def get_secret():
    """Retrieve database credentials from AWS Secrets Manager"""
    secret_name = "rds/sachin"
    region_name = "ap-southeast-1"

    try:
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        
        secret = get_secret_value_response['SecretString']
        logger.info(f"Successfully retrieved secret: {secret_name}")
        return json.loads(secret)
        
    except ClientError as e:
        logger.error(f"Failed to retrieve secret {secret_name}: {e}")
        raise e
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse secret JSON: {e}")
        raise e

def connect_to_rds():
    """Establish connection to RDS PostgreSQL database"""
    try:
        secret = get_secret()
        # Extract the necessary details from the secret
        db_host = secret['host']
        db_user = secret['username']
        db_password = secret['password']
        db_name = secret['database']
        db_port = secret['port']

        # Connect to the RDS instance
        connection = psycopg2.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            dbname=db_name,
            port=int(db_port),
            cursor_factory=RealDictCursor
        )
        logger.info(f"Successfully connected to database: {db_name}")
        return connection
        
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error connecting to database: {e}")
        return None

def is_supported_format(file_name: str) -> bool:
    """Check if file format is supported for audio processing"""
    if file_name.startswith('processing/') or file_name.startswith('processed_latest/') or file_name.startswith('processed/'):
        return False
    supported_formats = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
    return file_name.split('.')[-1].lower() in supported_formats

def move_s3_object(file_key: str, destination_key: str) -> bool:
    """Move S3 object from one key to another using copy and delete operation"""
    try:
        # First, copy the object to the new location
        s3_client.copy_object(
            Bucket=BUCKET_NAME,
            Key=destination_key,
            CopySource={
                'Bucket': BUCKET_NAME,
                'Key': file_key
            }
        )
        
        # Then delete the original object
        s3_client.delete_object(
            Bucket=BUCKET_NAME,
            Key=file_key
        )
        
        logger.info(f"Successfully moved S3 object from {file_key} to {destination_key}")
        return True
        
    except ClientError as e:
        logger.error(f"S3 client error moving object from {file_key} to {destination_key}: {e}")
        # Attempt to clean up if copy succeeded but delete failed
        try:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=destination_key)
        except:
            pass
        return False
    except Exception as e:
        logger.error(f"Unexpected error moving S3 object from {file_key} to {destination_key}: {e}")
        return False

def get_original_key_from_processed(processed_key: str) -> str:
    """Extract original key from processing/processed_latest/processed key"""
    if processed_key.startswith('processing/'):
        return processed_key.replace('processing/', '', 1)
    elif processed_key.startswith('processed_latest/'):
        return processed_key.replace('processed_latest/', '', 1)
    elif processed_key.startswith('processed/'):
        return processed_key.replace('processed/', '', 1)
    return processed_key


def check_if_file_is_processed(file_key: str) -> bool:
    """Checks if the file is already processed by querying the database"""
    connection = None
    try:
        connection = connect_to_rds()
        if not connection:
            logger.error("Failed to establish database connection")
            return False
        
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS CALLS (
                    call_id SERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_size BIGINT,
                    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
        # Commit using connection, not cursor
        connection.commit()
        
        with connection.cursor() as cursor:
            cursor.execute(
                """
                    SELECT COUNT(*) FROM CALLS WHERE file_name = %s
                """, (file_key,)
            )
            result = cursor.fetchone()
            count = result['count'] if result else 0
            
        logger.info(f"File {file_key} processed status: {count > 0}")
        return count > 0
        
    except psycopg2.Error as e:
        logger.error(f"Database error checking processed file {file_key}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking processed file {file_key}: {e}")
        return False
    finally:
        if connection:
            connection.close()

def s3_get_audio_file(key: str) -> bytes | None:
    """Fetch audio file from S3 and return as bytes (for transcription)."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        audio_bytes = response["Body"].read()
        return audio_bytes
    except botocore.exceptions.ClientError as e:
        print(f"Error fetching {key} from S3: {e}")
        return None
