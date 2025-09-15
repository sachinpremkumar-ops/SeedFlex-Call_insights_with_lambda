import json
import boto3
import urllib.request
import urllib.parse
import os
import hashlib
import time
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda function to invoke the FastAPI call insights endpoint
    
    Handles three types of events:
    1. Direct invocation with custom payload
    2. S3 event notifications (automatic triggers)
    3. SQS messages (from S3 events)
    
    Expected event structure for direct invocation:
    {
        "audio_file_key": "filename.mp3",
        "api_url": "http://your-fastapi-url:8000/process"  # Optional, defaults to localhost
    }
    
    SQS event structure (from S3):
    {
        "Records": [
            {
                "body": "{\"Records\": [{\"s3\": {\"bucket\": {\"name\": \"bucket-name\"}, \"object\": {\"key\": \"path/to/file.mp3\"}}}]}"
            }
        ]
    }
    """
    
    print(f"🚀 Lambda invoked with event: {json.dumps(event, indent=2)}")
    print(f"🔍 Context: {context}")
    
    try:
        print(f"🔍 Event type detection starting...")
        
        # Check if this is an SQS message (from S3 events)
        if 'Records' in event and event['Records'] and 'body' in event['Records'][0]:
            print(f"📨 Detected SQS message")
            # Handle SQS message containing S3 event
            sqs_record = event['Records'][0]
            print(f"📨 SQS record: {sqs_record}")
            
            s3_event = json.loads(sqs_record['body'])
            print(f"📨 Parsed S3 event: {json.dumps(s3_event, indent=2)}")
            
            if 'Records' in s3_event and s3_event['Records']:
                s3_record = s3_event['Records'][0]['s3']
                bucket_name = s3_record['bucket']['name']
                file_key = s3_record['object']['key']
                file_size = s3_record['object'].get('size', 0)
                
                print(f"📨 SQS Message received: Bucket={bucket_name}, Key={file_key}, Size={file_size}")
                
                # Skip processing if file is already in processed folders
                if file_key.startswith('processed_latest/') or file_key.startswith('processing/'):
                    print(f"⚠️ Skipping already processed file: {file_key}")
                    return {
                        'statusCode': 200,
                        'body': json.dumps({
                            'message': f'File {file_key} already processed, skipping'
                        })
                    }
                
                audio_file_key = file_key
                print(f"🎯 Processing file from SQS: {audio_file_key}")
            else:
                print(f"❌ No S3 records found in SQS message")
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'No S3 records found in SQS message'})
                }
                
        # Check if this is a direct S3 event notification (legacy)
        elif 'Records' in event and event['Records'] and 's3' in event['Records'][0]:
            # Handle direct S3 event notification
            s3_record = event['Records'][0]['s3']
            bucket_name = s3_record['bucket']['name']
            file_key = s3_record['object']['key']
            file_size = s3_record['object'].get('size', 0)
            
            # Skip processing if file is already in processed folders
            if file_key.startswith('processed_latest/') or file_key.startswith('processing/'):
                print(f"⚠️ Skipping already processed file: {file_key}")
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': f'File {file_key} already processed, skipping'
                    })
                }
            
            audio_file_key = file_key
            print(f"📁 Direct S3 Event detected: Bucket={bucket_name}, Key={file_key}, Size={file_size}")
        else:
            # Handle direct invocation
            messages = event.get('messages', '')
            audio_file_key = event.get('audio_file_key')
            
            if not audio_file_key:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': 'Missing required field: audio_file_key'
                    })
                }
            
            # If no messages provided, use audio_file_key as the message
            if not messages:
                messages = f"Process audio file: {audio_file_key}"
        
        # Get API URL from event or environment variable
        api_url = event.get('api_url', os.environ.get('FASTAPI_URL', 'http://127.0.0.1:8001/process'))
        
        # Prepare the request payload
        payload = {
            "audio_file_key": audio_file_key,  # Optional: provide custom message, defaults to audio_file_key
        }
        
        print(f"Sending payload to {api_url}: {payload}")
        
        # Make the request to your FastAPI endpoint
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            api_url,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                response_data = json.loads(response.read().decode('utf-8'))
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'success': True,
                        'data': response_data,
                        'message': 'Call insights processed successfully'
                    })
                }
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            return {
                'statusCode': e.code,
                'body': json.dumps({
                    'error': f'API request failed with status {e.code}',
                    'details': error_body
                })
            }
            
    except urllib.error.URLError as e:
        if 'timeout' in str(e).lower():
            return {
                'statusCode': 504,
                'body': json.dumps({
                    'error': 'Request timeout - processing took too long'
                })
            }
        else:
            return {
                'statusCode': 503,
                'body': json.dumps({
                    'error': f'Unable to connect to FastAPI service: {str(e)}'
                })
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        }
