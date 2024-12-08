import json
import base64
import boto3
import logging
from datetime import datetime

# Create an S3 client
s3_client = boto3.client('s3')

# Define the S3 bucket name
bucket_name = 'db67'

def lambda_handler(event, context):
    # Log the raw Kinesis data
    print("Raw Kinesis Data:", event)
    
    # Process each record from the Kinesis stream
    for record in event['Records']:
        # Decode the Kinesis record's payload
        payload = json.loads(base64.b64decode(record['kinesis']['data']).decode('utf-8'))
        
        # Print the decoded payload
        print("Decoded Payload:", payload)
        
        # Generate a unique filename using the timestamp or another unique identifier
        filename = f"{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}_data.json"
        
        # Convert the payload back to JSON for storing in S3
        s3_data = json.dumps(payload)
        
        # Upload the data to S3
        try:
            response = s3_client.put_object(
                Bucket=bucket_name,
                Key=filename,
                Body=s3_data,
                ContentType='application/json'
            )
            print(f"Successfully added data to S3: {filename}")
        except Exception as e:
            print(f"Error uploading to S3: {e}")
