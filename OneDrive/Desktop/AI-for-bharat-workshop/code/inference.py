import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

def summarize_text(text):
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
    )
    
    prompt = f"Summarize this in 2–3 lines:\n\n{text}"
    
    body = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.3
    }

    response = client.invoke_model(
        modelId="amazon.nova-lite-v1:0",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["output_text"]
