from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

api_token = os.getenv("HUGGING_FACE_API_TOKEN")

client = InferenceClient(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    token=api_token,
)

def parse_with_huggingface(dom_chunks, parse_description):
    parsed_results = []
    for chunk in dom_chunks:
        messages = [{"role": "user", "content": f"Extract information: {parse_description}\nContent: {chunk}"}]
        
        for message in client.chat_completion(
            messages=messages,
            max_tokens=500,
            stream=False,  
        ):

            parsed_results.append(message.choices[0].delta.content)
    
    return "\n".join(parsed_results)

