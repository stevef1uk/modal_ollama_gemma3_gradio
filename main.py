"""
Modal-based Ollama Server with Gemma 3 27B Model

This module implements a high-performance Ollama server using Modal cloud infrastructure
and the Gemma 3 27B model. It includes:

1. Persistent model caching using Modal volumes
2. GPU-accelerated inference using H100
3. Automatic container lifecycle management
4. Efficient request handling with idle timeout

Key Components:
- OllamaService: Main class that manages the Ollama service lifecycle
  - start(): Initializes and validates the Ollama service
  - stop(): Cleanly shuts down the service
  - serve(): Handles inference requests

Usage:
1. Deploy the server:
   modal deploy main.py

2. Make requests:
   export API_URL="https://[username]--[app-name]-validate.modal.run"
   
   curl -X POST "${API_URL}" \
        -H "Content-Type: application/json" \
        -d '{
          "prompt": "Your prompt here",
          "temperature": 0.7
        }'

Features:
- Uses Gemma 3 27B model for high-quality responses
- Automatic model caching between requests
- Container shuts down after 5 minutes of inactivity
- GPU acceleration using H100
- Persistent model storage using Modal volumes

Dependencies:
- Modal
- Ollama
- Python requests
"""

# DeepSeek LLM Server with llama.cpp
#
# Authors:
# - Original implementation
# - Steven Fisher (stevef1uk@gmail.com) + Cursor :-)
#
# This implementation provides a FastAPI server running DeepSeek-R1 language model
# using llama.cpp backend. It features:
#
# - GPU-accelerated inference using CUDA
# - Bearer token authentication
# - Automatic model downloading and caching
# - GGUF model file merging
# - Swagger UI documentation
#
# Key Components:
#
# 1. Infrastructure Setup:
#    - Uses Modal for serverless deployment
#    - CUDA 12.4.0 with development toolkit
#    - Python 3.12 environment
#
# 2. Model Configuration:
#    - Gemma3 27B model
#
# 3. Server Features:
#    - FastAPI-based REST API
#    - Bearer token authentication
#    - Interactive documentation at /docs endpoint
#    - Configurable context length and batch size
#    - Flash attention support
#
# Hardware Requirements:
#    - 1x NVIDIA H100-80GB GPU
#    - Supports concurrent requests (if configured to do that)
#
#
# 1. Deploy the server:
#    modal deploy main.py
#
# 2. Test the deployment:
#    export API_URL="https://[username]--[app-name]-validate.modal.run"
#    export API_TOKEN="your-token-from-logs"
#    
#    curl -X POST "${API_URL}/v1/completions" \
#      -H "Content-Type: application/json" \
#      -H "Authorization: Bearer ${API_TOKEN}" \
#      -d '{
#        "prompt": "What is the capital of France?",
#        "max_tokens": 100,
#        "temperature": 0.7
#      }'
#
# Authentication:
# All API endpoints (except documentation) require Bearer token authentication
# Example:
# curl -H "Authorization: Bearer your-token" \
#   https://[username]--[app-name]-validate.modal.run/v1/completions \
#   -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'
#
# Model Settings:
# - Context length (n_ctx): 4096
# - Batch size (n_batch): 128
# - Thread count (n_threads): 12
# - GPU Layers: All (-1)
# - Flash Attention: Enabled
#
# Note: 
# 1. The server includes automatic redirection from root (/) to documentation (/docs)2
# 2. GPU costs vary by type and count - monitor usage accordingly
# 3. For your first request the model will be downloaded and cached, this may take a while  
# 4. The API will go to sleep and stop and 5 minutes of inactivity, this is to save costs
# 5. The API will wake up and start serving requests again when a request comes in


from __future__ import annotations

import glob
import subprocess
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
import modal
from modal import Secret
import requests
import shutil
import psutil

# Constants for CUDA setup
cuda_version = "12.4.0"  # Latest stable CUDA version
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create Modal application
app = modal.App("gemma-api")

# Constants
MINUTES = 60
INFERENCE_TIMEOUT = 15 * MINUTES
MERGE_TIMEOUT = 60 * MINUTES  # Increase to 1 hour to be safe
MODELS_DIR = "/gemma"
cache_dir = "/root/.cache/gemma"

# System memory for H100-80GB
SYSTEM_MEMORY = 131072  # 128GB of system memory

# After the constants and before any function definitions
def get_gpu_config():
    """Helper function to get the correct GPU configuration"""
    gpu_type = os.environ.get("gpu_type", "H100")
    gpu_count = int(os.environ.get("gpu_count", "1"))  # Changed from 3 to 1 for Gemma
    
    gpu_map = {
        "H100": "H100",
        "A100": "A100-40GB",
        "L40": "L40-48GB",
        "A10G": "A10G-24GB",
    }
    
    if gpu_type not in gpu_map:
        print(f"⚠️ Warning: Unknown GPU type {gpu_type}, falling back to H100")
        gpu_type = "H100"
    
    gpu_spec = f"{gpu_map[gpu_type]}:{gpu_count}"
    print(f"🖥️ Using GPU configuration: {gpu_spec}")
    return gpu_spec

# Move this up near the other constants and imports
TOKEN = secrets.token_urlsafe(32)
print(f"🔑 Your API token is: {TOKEN}")
print("\n To create/update the Modal secret, run this command:")
print(f"modal secret create MODAL_SECRET_LLAMA_CPP_API_KEY llamakey={TOKEN}")

# Create a temporary secret for this run
secret = Secret.from_dict({"TOKEN": TOKEN})
print(f"🔒 Created Modal secret with token")

# Add HuggingFace token setup near the top with other secrets
hf_secret = Secret.from_name("hf-secret")  # Use your existing Modal secret

# Create persistent volume for storing Ollama models between container restarts
model_cache = modal.Volume.from_name("ollama-model-cache", create_if_missing=True)

# Base image configuration with required dependencies
base_image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install("requests", "psutil")
    .run_commands([
        "apt-get update && apt-get install -y curl",
        "curl -fsSL https://ollama.ai/install.sh | sh",
        "chmod +x /usr/local/bin/ollama"
    ])
    .env({"CUDA_VISIBLE_DEVICES": "0"})
)

# 
# Use Modal's base Python image
vllm_image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install("requests")
    .run_commands([
        "pip install --no-cache-dir requests fastapi sse_starlette pydantic uvicorn[standard] python-multipart starlette-context pydantic-settings psutil",
        "apt-get update && apt-get install -y curl gpg pciutils lshw && rm -rf /var/lib/apt/lists/*",
        "curl -fsSL https://ollama.ai/install.sh | sh",
        "chmod +x /usr/local/bin/ollama"
    ])
    .env({"CUDA_VISIBLE_DEVICES": "0"})
)

# Update the validation image to include psutil
validation_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "requests", "uvicorn[standard]", "psutil")
)

# Add a flag to track if model has been downloaded
model_downloaded = False

# Global variable to store the LLM instance
_llm = None

# Common timeout settings
TIMEOUT = 1000  # 1000 seconds
IDLE_TIMEOUT = 300  # 5 minutes

MODEL_NAME = "gemma3:27b"

@app.cls(
    image=base_image,
    gpu="H100",
    volumes={"/root/.ollama": model_cache},
    container_idle_timeout=300,  # 5 minutes idle timeout
    timeout=1800  # 30 minutes timeout to allow for model pulling
)
class OllamaService:
    @modal.enter()
    def start(self):
        """Start the Ollama service when container starts."""
        print("Starting Ollama service...")
        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for service with retries
        max_retries = 6
        for i in range(max_retries):
            try:
                import requests
                health_check = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
                if health_check.status_code == 200:
                    print(f"Ollama service status: {health_check.status_code}")
                    models = health_check.json().get('models', [])
                    print(f"Available models: {models}")
                    
                    # Check if our model needs to be pulled
                    if not any(m['name'] == MODEL_NAME for m in models):
                        print(f"Pulling model {MODEL_NAME}...")
                        result = subprocess.run(
                            ["ollama", "pull", MODEL_NAME],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode != 0:
                            raise Exception(f"Model pull failed: {result.stderr}")
                        print("Model pull completed successfully!")
                    return
            except Exception as e:
                if i < max_retries - 1:
                    print(f"Waiting for Ollama service (attempt {i+1}/{max_retries})...")
                    time.sleep(5)
                else:
                    raise Exception(f"Failed to start Ollama service: {str(e)}")

    @modal.exit()
    def stop(self):
        """Stop the Ollama service when the container stops."""
        if hasattr(self, 'ollama_process'):
            print("Shutting down Ollama service...")
            self.ollama_process.terminate()
            self.ollama_process.wait(timeout=30)
            print("Ollama service terminated")

    @modal.method()
    async def serve(self, request_data: dict):
        """Handle inference requests."""
        print("\nSending inference request...")
        try:
            import requests  # Import here to ensure it's available
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": request_data["prompt"],
                    "stream": request_data.get("stream", False),
                    "options": {
                        "temperature": request_data.get("temperature", 0.7),
                        "top_p": 0.9,
                        "top_k": 40,
                    }
                },
                timeout=1800
            )
            response.raise_for_status()
            print("Response received successfully")
            return response.json()
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise

@app.function(
    image=validation_image,
    gpu=None,
    timeout=3600,
    container_idle_timeout=300,
    secrets=[secret],
)
@modal.asgi_app()
def validate():
    """Initial validation endpoint that gates access to the LLM server"""
    import json
    import os
    
    async def app(scope, receive, send):
        """Simple ASGI application"""
        if scope["type"] != "http":
            return
            
        # Get the request method and path
        method = scope["method"]
        path = scope["path"]
        
        if method != "POST" or path != "/v1/completions":
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": "Not found"}).encode()
            })
            return
            
        # Check authorization
        headers = dict(scope["headers"])
        auth_header = headers.get(b"authorization", b"").decode()
        
        if not auth_header:
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer")
                ]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": "No authentication token provided"}).encode()
            })
            return
            
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer" or token != os.environ["TOKEN"]:
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer")
                    ]
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps({"detail": "Invalid authentication token"}).encode()
                })
                return
                
        except Exception:
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer")
                ]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": "Invalid authentication header"}).encode()
            })
            return
            
        # Read request body
        message = await receive()
        body = message.get('body', b'')
        
        # Handle more body chunks if any
        while message.get('more_body', False):
            message = await receive()
            body += message.get('body', b'')
            
        try:
            request_data = json.loads(body)
            print(f"Received request data: {request_data}")
            
            if "prompt" not in request_data:
                await send({
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json")]
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps({"detail": "prompt is required"}).encode()
                })
                return
                
            # Prepare request for serve
            processed_data = {
                "prompt": request_data["prompt"],
                "max_tokens": request_data.get("max_tokens", 100),
                "temperature": request_data.get("temperature", 0.7),
                "stream": request_data.get("stream", False)  # Use the requested stream value
            }
            
            print(f"Calling serve with data: {processed_data}")
            
            if processed_data["stream"]:
                # Streaming response
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")]
                })
                
                async for chunk in OllamaService.serve.remote_gen(processed_data):
                    await send({
                        "type": "http.response.body",
                        "body": json.dumps(chunk).encode(),
                        "more_body": True
                    })
                
                # Send final empty chunk
                await send({
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False
                })
            else:
                # Non-streaming response
                response = OllamaService.serve.remote(processed_data)
                
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")]
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps(response).encode(),
                    "more_body": False
                })
            
        except json.JSONDecodeError:
            await send({
                "type": "http.response.start",
                "status": 400,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": "Invalid JSON in request body"}).encode()
            })
        except Exception as e:
            print(f"Error: {str(e)}")
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": str(e)}).encode()
            })
    
    return app

def create_model():
    """Pull the model if it doesn't exist."""
    print("\nPulling Gemma 27B model from Ollama...")
    process = subprocess.Popen(
        ["ollama", "pull", "gemma:27b"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    while process.poll() is None:
        stdout = process.stdout.readline()
        if stdout:
            print(f"Output: {stdout.strip()}")
        stderr = process.stderr.readline()
        if stderr:
            print(f"Error: {stderr.strip()}")
        time.sleep(1)
    
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Model pull failed: {stderr}")
    
    print("Model pull completed successfully!")
    return True

@app.local_entrypoint()
def main(skip_download: bool = False):
    """Run initialization with optional download skip"""
    print(f"🚀 Starting initialization with skip_download={skip_download}")
    return OllamaService.serve.remote(skip_download)

if __name__ == "__main__":
    print("You can now deploy this with: modal deploy main.py")