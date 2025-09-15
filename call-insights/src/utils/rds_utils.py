import boto3
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from botocore.exceptions import ClientError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# def get_secret():
#     """Retrieve database credentials from AWS Secrets Manager"""
#     secret_name = "rds/sachin"
#     region_name = "ap-southeast-1"

#     try:
#         # Create a Secrets Manager client
#         session = boto3.session.Session()
#         client = session.client(
#             service_name='secretsmanager',
#             region_name=region_name
#         )

#         get_secret_value_response = client.get_secret_value(
#             SecretId=secret_name
#         )
        
#         secret = get_secret_value_response['SecretString']
#         logger.info(f"Successfully retrieved secret: {secret_name}")
#         return json.loads(secret)
        
#     except ClientError as e:
#         logger.error(f"Failed to retrieve secret {secret_name}: {e}")
#         raise e
#     except json.JSONDecodeError as e:
#         logger.error(f"Failed to parse secret JSON: {e}")
#         raise e

def connect_to_rds():
    """Establish connection to RDS PostgreSQL database"""
    try:
        # Direct database connection without using secrets manager
        db_host = "experimentalrds.c548sgyk8ab2.ap-southeast-1.rds.amazonaws.com"
        db_user = "sachin_premkumar"
        db_password = "Sachin_2025@01!"
        db_name = "postgres"
        db_port = "5432"

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
