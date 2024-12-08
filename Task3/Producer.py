import boto3
import json
import time
from random import randint
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Initialize the Kinesis client
kinesis = boto3.client('kinesis', region_name='us-east-1')  # Replace with your AWS region
stream_name = 'RealTimeStream'  # Replace with your Kinesis stream name

def generate_data():
    """
    Simulates sensor data and formats it as a JSON object.
    """
    data = {
        "timestamp": int(time.time()),  # Current timestamp
        "sensor_id": "sensor-01",  # Sensor identifier
        "temperature": round(randint(20, 30) + randint(0, 99) / 100, 2),  # Random temperature between 20.00 and 30.99
        "humidity": round(randint(30, 80) + randint(0, 99) / 100, 2)  # Random humidity between 30.00 and 80.99
    }
    return json.dumps(data)

def send_to_kinesis():
    """
    Sends the generated JSON data to Kinesis stream.
    """
    while True:
        try:
            # Generate and format data
            payload = generate_data()
            # Log the data locally to the console
            logger.info(f"Sending data to Kinesis: {payload}")

            # Send data to Kinesis stream
            response = kinesis.put_record(
                StreamName=stream_name,
                Data=payload,  # The data to send (JSON format)
                PartitionKey="partition-key"  # Partition key (can be sensor ID for real applications)
            )
            # Log the response from Kinesis
            logger.info(f"Kinesis Response: {response}")
        except Exception as e:
            logger.error(f"Error sending data to Kinesis: {e}")
        
        # Wait before sending the next data point (simulate real-time data stream)
        time.sleep(2)

if __name__ == "__main__":
    send_to_kinesis()
