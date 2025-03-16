"""
Gradio GUI for Ollama Server with Multiple Model Support

This script deploys a Gradio GUI on Modal that interacts with a remote Ollama server.
Author: Steven Fisher (stevef@gmail.com)

Description: This script sets up a Gradio interface on Modal, which communicates
             with a separate Ollama server. It requires the following secrets:
             - MODAL_PROXY_TOKEN_ID: Token ID for Modal proxy authentication
             - MODAL_PROXY_TOKEN_SECRET: Token secret for Modal proxy authentication
             - gradio_app_access_key: Access key for securing the Gradio app.
             - llama_server_url: URL of the remote Ollama server.
"""

from fastapi import FastAPI
import gradio as gr
from gradio.routes import mount_gradio_app
import uvicorn
import os
import requests
import json
import modal
import time

# Create Modal app
modal_app = modal.App("ollama-gradio-interface")

# Create Modal secret references
server_url_secret = modal.Secret.from_name("llama_server_url")
proxy_token_id = modal.Secret.from_name("MODAL_PROXY_TOKEN_ID")
proxy_token_secret = modal.Secret.from_name("MODAL_PROXY_TOKEN_SECRET")
gradio_access_secret = modal.Secret.from_name("gradio_app_access_key")

# Create a persistent FastAPI app
web_app = FastAPI()

# List of popular Ollama models
OLLAMA_MODELS = [
    "gemma3:27b",
    "llama3:8b",
    "llama3:70b",
    "phi3:14b",
    "mistral:7b",
    "mixtral:8x7b",
    "codellama:70b",
    "llama2:70b",
    "orca2:13b",
    "vicuna:13b"
]

# Get token values from secrets at startup
TOKEN_ID = None
TOKEN_SECRET = None

@web_app.on_event("startup")
async def startup_event():
    """Initialize token values from secrets at startup."""
    global TOKEN_ID, TOKEN_SECRET
    
    # Try to get token values from environment variables
    TOKEN_ID = os.environ.get("MODAL_SECRET_MODAL_PROXY_TOKEN_ID")
    if not TOKEN_ID:
        # Try alternate environment variable names
        TOKEN_ID = os.environ.get("token_id")
    
    TOKEN_SECRET = os.environ.get("MODAL_SECRET_MODAL_PROXY_TOKEN_SECRET")
    if not TOKEN_SECRET:
        # Try alternate environment variable names
        TOKEN_SECRET = os.environ.get("token_secret")
    
    print(f"Tokens initialized at startup: ID={bool(TOKEN_ID)}, Secret={bool(TOKEN_SECRET)}")
    
    # Print all environment variables for debugging
    print("All environment variables:")
    for key in sorted(os.environ.keys()):
        print(f"  {key}")

def chat_with_llm(access_key: str, model: str, message: str):
    """Chat function that uses the LLM service."""
    global TOKEN_ID, TOKEN_SECRET
    
    print(f"Received chat request with model: {model} and message: {message}")
    
    # Retrieve the expected access key from environment variables
    expected_access_key = os.environ.get("MODAL_SECRET_GRADIO_APP_ACCESS_KEY")
    if access_key != expected_access_key:
        yield "Error: Invalid access key."
        return

    # Access the server URL
    server_url = os.environ.get("LLAMA_SERVER_URL")
    if not server_url:
        yield "Error: Server URL not found in environment variables"
        return
    
    # Check if tokens are available
    if not TOKEN_ID or not TOKEN_SECRET:
        # Try to get token values from environment variables again
        TOKEN_ID = os.environ.get("MODAL_SECRET_MODAL_PROXY_TOKEN_ID")
        if not TOKEN_ID:
            TOKEN_ID = os.environ.get("token_id")
        
        TOKEN_SECRET = os.environ.get("MODAL_SECRET_MODAL_PROXY_TOKEN_SECRET")
        if not TOKEN_SECRET:
            TOKEN_SECRET = os.environ.get("token_secret")
        
        if not TOKEN_ID or not TOKEN_SECRET:
            yield "Error: Modal proxy authentication tokens not available. Please check your secrets configuration."
            return
    
    headers = {
        "Content-Type": "application/json",
        "Modal-Key": TOKEN_ID,
        "Modal-Secret": TOKEN_SECRET
    }
    
    print(f"Using authentication with tokens from secrets")
    
    enhanced_prompt = f"Please provide a clear, concise answer to this question: {message}"
    
    # For non-streaming request
    payload = {
        "prompt": enhanced_prompt,
        "temperature": 0.7,
        "model": model,  # Use the selected model
        "stream": False
    }

    try:
        print("Making request to server...")
        print(f"Using server URL: {server_url}")
        print(f"Using model: {model}")
        
        # First yield a message indicating the model is being loaded
        yield f"Loading {model}... This may take a moment if the model isn't already cached."
        
        response = requests.post(
            server_url,
            headers=headers,
            json=payload,
            timeout=600  # Increased timeout for model loading
        )
        print(f"Response status code: {response.status_code}")
        
        response.raise_for_status()
        
        try:
            json_response = response.json()
            print(f"Response received: {json_response}")
            
            # Handle Ollama response format
            if 'response' in json_response:
                text = json_response['response'].strip()
                if text:
                    # Simulate streaming by yielding chunks of the text
                    words = text.split()
                    accumulated = ""
                    chunk_size = 3  # Number of words per chunk
                    
                    for i in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[i:i + chunk_size])
                        accumulated += chunk + ' '
                        yield accumulated.strip()
                        time.sleep(0.1)  # Small delay between chunks
                    
                    # Ensure we yield the complete text at the end
                    if accumulated.strip() != text:
                        yield text
                else:
                    yield "No text in response"
            else:
                yield "No response field found in server response"
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            yield f"Error decoding response: {str(e)}"
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error: {str(e)}"
        print(error_msg)
        yield error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        yield error_msg

# Create the Gradio interface
demo = gr.Interface(
    fn=chat_with_llm,
    inputs=[
        gr.Textbox(label="Access Key", type="password"),
        gr.Dropdown(
            choices=OLLAMA_MODELS,
            value="gemma3:27b",  # Default model
            label="Select Model",
            info="Choose the Ollama model to use for generating responses"
        ),
        gr.Textbox(
            label="Enter your message", 
            lines=4,
            placeholder="Type your question or prompt here..."
        )
    ],
    outputs=gr.Textbox(
        label="Response", 
        interactive=False, 
        lines=15,
        max_lines=30,
        autoscroll=False,
        show_copy_button=True,
        container=True,
        scale=2
    ),
    title="Ollama LLM Chat Interface",
    description="Enter the access key, select a model, and type your message to chat with various Ollama models.",
    live=False,
    flagging_mode="never"
)

# Mount Gradio app to FastAPI
gradio_app = mount_gradio_app(
    app=web_app,
    blocks=demo,
    path="/"
)

@modal_app.function(
    secrets=[
        {"name": "MODAL_PROXY_TOKEN_ID", "mount_path": "/secrets/token_id"},
        {"name": "MODAL_PROXY_TOKEN_SECRET", "mount_path": "/secrets/token_secret"},
        server_url_secret,
        gradio_access_secret
    ]
)
@modal.asgi_app()
def serve():
    """Return the persistent ASGI app"""
    # Try to read token values from mounted secret files
    global TOKEN_ID, TOKEN_SECRET
    
    try:
        with open("/secrets/token_id", "r") as f:
            TOKEN_ID = f.read().strip()
        
        with open("/secrets/token_secret", "r") as f:
            TOKEN_SECRET = f.read().strip()
        
        print(f"Tokens loaded from mounted secret files: ID={bool(TOKEN_ID)}, Secret={bool(TOKEN_SECRET)}")
    except Exception as e:
        print(f"Error reading token files: {str(e)}")
    
    return web_app

if __name__ == "__main__":
    uvicorn.run(web_app, host="0.0.0.0", port=8000)