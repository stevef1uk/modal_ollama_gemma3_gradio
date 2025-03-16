"""
Modal-based Ollama Server with Multiple LLM Support

This module implements a high-performance Ollama server using Modal cloud infrastructure
with support for multiple language models. It includes:

1. Persistent model caching using a Modal volume
2. Configurable GPU acceleration (H100, A100, L40, A10G)
3. Automatic container lifecycle management
4. Efficient request handling with configurable idle timeout
5. Support for multiple LLM models (gemma3:27b, llama3:8b, phi3:14b, mistral:7b, etc.)
6. Configurable container scaling limits to control costs

Key Features:
- Persistent Ollama service that stays running between requests
- Model caching to avoid reloading models for each request
- Configurable GPU type and count via environment variables or command-line arguments
- Configurable default model via environment variables or command-line arguments
- Configurable maximum container limit to control parallelism and costs
- Configurable idle timeout for automatic container shutdown to save costs
- Modal proxy authentication for secure API access

Usage:
1. Deploy the server:
   modal deploy main.py
   
   # Or with custom configuration:
   python main.py --gpu-type H100 --model llama3:8b --max-containers 1 --idle-timeout 600 --deploy

2. Make requests:
   export API_URL="https://[username]--ollama-api-api.modal.run"
   export TOKEN_ID="your-modal-token-id"
   export TOKEN_SECRET="your-modal-token-secret"
   
   curl -X POST "${API_URL}" \
        -H "Content-Type: application/json" \
        -H "Modal-Key: $TOKEN_ID" \
        -H "Modal-Secret: $TOKEN_SECRET" \
        -d '{
          "prompt": "Your prompt here",
          "temperature": 0.7,
          "model": "gemma3:27b"
        }'

Environment Variables:
- GPU_TYPE: Type of GPU to use (H100, A100, L40, A10G)
- GPU_COUNT: Number of GPUs per container
- MODEL_NAME: Default model to use
- MAX_CONTAINERS: Maximum number of containers for parallel processing
- IDLE_TIMEOUT: Idle timeout in seconds before container shutdown

Dependencies:
- Modal
- Ollama
- Python requests
- FastAPI
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
#    export API_URL="https://[username]--ollama-api-api.modal.run"
#    
#    curl -X POST "${API_URL}" \
#      -H "Content-Type: application/json" \
#      -H "Modal-Key: $TOKEN_ID" \
#      -H "Modal-Secret: $TOKEN_SECRET" \
#      -d '{
#        "prompt": "What is the capital of France?",
#        "temperature": 0.7
#      }'
#
# Authentication:
# All API endpoints require Modal proxy authentication
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
# 4. The API will go to sleep after the configured idle timeout (default: 5 minutes)
# 5. The API will wake up and start serving requests again when a request comes in
# 6. Container scaling is limited to control costs (default: max 2 containers)


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
from pydantic import BaseModel
from typing import Optional
import argparse

# Constants for CUDA setup
cuda_version = "12.4.0"  # Latest stable CUDA version
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create Modal application
app = modal.App("ollama-api")

# Time constants
MINUTES = 60
INFERENCE_TIMEOUT = 15 * MINUTES
MERGE_TIMEOUT = 60 * MINUTES  # Increase to 1 hour to be safe
DEFAULT_IDLE_TIMEOUT = 5 * MINUTES  # 5 minutes (default container idle timeout)

# File paths
MODELS_DIR = "/gemma"
cache_dir = "/root/.cache/gemma"

# System memory for H100-80GB
SYSTEM_MEMORY = 131072  # 128GB of system memory

# Default GPU configuration
DEFAULT_GPU_TYPE = "A10G"    #Default to the cheapest GPU you can always configure a bigger one e.g. H100
DEFAULT_GPU_COUNT = 1

# Add this with the other default configurations
DEFAULT_MAX_CONTAINERS = 2  # Default maximum number of containers

def get_gpu_config():
    """
    Get GPU configuration from environment variables or use defaults.
    
    This function reads GPU_TYPE and GPU_COUNT from environment variables,
    validates them against supported options, and returns a properly formatted
    GPU specification string for Modal.
    
    Returns:
        str: GPU specification string (e.g., "H100", "A100", "L40", "A10G")
    """
    gpu_type = os.environ.get("GPU_TYPE", DEFAULT_GPU_TYPE)
    gpu_count_str = os.environ.get("GPU_COUNT", str(DEFAULT_GPU_COUNT))
    
    try:
        gpu_count = int(gpu_count_str)
    except ValueError:
        print(f"⚠️ Warning: Invalid GPU count '{gpu_count_str}', using default: {DEFAULT_GPU_COUNT}")
        gpu_count = DEFAULT_GPU_COUNT
    
    # Validate GPU type against supported options
    # Updated to match Modal's expected format
    valid_gpu_types = ["H100", "A100", "L40", "A10G"]
    
    if gpu_type not in valid_gpu_types:
        print(f"⚠️ Warning: Unknown GPU type '{gpu_type}', falling back to {DEFAULT_GPU_TYPE}")
        gpu_type = DEFAULT_GPU_TYPE
    
    # Format the GPU specification string
    gpu_spec = gpu_type
    if gpu_count > 1:
        gpu_spec += f":{gpu_count}"
    
    print(f"🖥️ Using GPU configuration: {gpu_spec}")
    return gpu_spec

# Create persistent volume for storing Ollama models between container restarts
model_cache = modal.Volume.from_name("ollama-model-cache", create_if_missing=True)

# Base image configuration with required dependencies
base_image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install("requests", "psutil", "fastapi[standard]", "uvicorn")
    .run_commands([
        "apt-get update && apt-get install -y curl",
        "curl -fsSL https://ollama.ai/install.sh | sh",
        "chmod +x /usr/local/bin/ollama"
    ])
    .env({"CUDA_VISIBLE_DEVICES": "0"})
)

# Common timeout settings
TIMEOUT = 1000  # 1000 seconds

# Default model, can be overridden via environment variable
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma3:27b")

# Add a new function to get the max containers configuration
def get_max_containers():
    """
    Get maximum container limit from environment variables or use default.
    
    This function reads MAX_CONTAINERS from environment variables,
    validates it, and returns the container limit.
    
    Returns:
        int: Maximum number of containers (default: 2)
    """
    max_containers_str = os.environ.get("MAX_CONTAINERS", str(DEFAULT_MAX_CONTAINERS))
    
    try:
        max_containers = int(max_containers_str)
        if max_containers < 1:
            print(f"⚠️ Warning: Invalid container limit '{max_containers_str}', using default: {DEFAULT_MAX_CONTAINERS}")
            max_containers = DEFAULT_MAX_CONTAINERS
    except ValueError:
        print(f"⚠️ Warning: Invalid container limit '{max_containers_str}', using default: {DEFAULT_MAX_CONTAINERS}")
        max_containers = DEFAULT_MAX_CONTAINERS
    
    print(f"🔢 Maximum containers: {max_containers}")
    return max_containers

# Add a function to get the idle timeout configuration
def get_idle_timeout():
    """
    Get idle timeout from environment variables or use default.
    
    This function reads IDLE_TIMEOUT from environment variables,
    validates it, and returns the idle timeout in seconds.
    
    Returns:
        int: Idle timeout in seconds (default: 300 seconds / 5 minutes)
    """
    idle_timeout_str = os.environ.get("IDLE_TIMEOUT", str(DEFAULT_IDLE_TIMEOUT))
    
    try:
        idle_timeout = int(idle_timeout_str)
        if idle_timeout < 60:  # Minimum 1 minute
            print(f"⚠️ Warning: Idle timeout too short '{idle_timeout_str}', using default: {DEFAULT_IDLE_TIMEOUT}")
            idle_timeout = DEFAULT_IDLE_TIMEOUT
    except ValueError:
        print(f"⚠️ Warning: Invalid idle timeout '{idle_timeout_str}', using default: {DEFAULT_IDLE_TIMEOUT}")
        idle_timeout = DEFAULT_IDLE_TIMEOUT
    
    print(f"⏱️ Idle timeout: {idle_timeout} seconds ({idle_timeout/60:.1f} minutes)")
    return idle_timeout

@app.function(
    image=base_image,
    gpu=get_gpu_config(),
    volumes={"/root/.ollama": model_cache},
    scaledown_window=get_idle_timeout(),
    timeout=1800,
    max_containers=get_max_containers()
)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def api(request_data: dict):
    """
    API endpoint that serves LLM inference requests through Ollama.
    
    This function:
    1. Maintains a persistent Ollama service between requests
    2. Tracks which models are already loaded to avoid redundant loading
    3. Dynamically loads requested models if they're not already available
    4. Processes inference requests with configurable parameters
    
    Args:
        request_data (dict): Request data containing:
            - prompt (str): The text prompt to send to the model
            - temperature (float, optional): Sampling temperature (default: 0.7)
            - model (str, optional): Model name to use (default: from environment)
    
    Returns:
        dict: Response containing:
            - model (str): The model used for inference
            - created (int): Unix timestamp of when the response was created
            - response (str): The model's response text
            - done (bool): Whether the generation is complete
            - error (str, optional): Error message if something went wrong
    """
    # Global variables to persist across requests within the same container
    global ollama_process, loaded_models
    
    # Initialize globals if they don't exist
    if 'ollama_process' not in globals():
        ollama_process = None
    if 'loaded_models' not in globals():
        loaded_models = set()
    
    try:
        # Extract data from the request
        prompt = request_data.get("prompt", "Hello")
        temperature = request_data.get("temperature", 0.7)
        model_name = request_data.get("model", MODEL_NAME)
        
        print(f"Received request with prompt: {prompt}")
        print(f"Using model: {model_name}")
        
        # Start Ollama service if not already running
        if ollama_process is None:
            print("Starting Ollama service...")
            ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for service with retries
            max_retries = 6
            for i in range(max_retries):
                try:
                    health_check = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
                    if health_check.status_code == 200:
                        print(f"Ollama service status: {health_check.status_code}")
                        models = health_check.json().get('models', [])
                        print(f"Available models: {models}")
                        
                        # Track which models are already loaded
                        for model in models:
                            loaded_models.add(model.get('name'))
                        
                        break
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"Waiting for Ollama service (attempt {i+1}/{max_retries})...")
                        time.sleep(5)
                    else:
                        raise Exception(f"Failed to start Ollama service: {str(e)}")
        
        # Ensure requested model is loaded
        if model_name not in loaded_models:
            print(f"Pulling model {model_name}...")
            try:
                result = subprocess.run(
                    ["ollama", "pull", model_name],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    raise Exception(f"Model pull failed: {result.stderr}")
                print("Model pull completed successfully!")
                loaded_models.add(model_name)
            except Exception as e:
                print(f"Error pulling model: {str(e)}")
                raise
        else:
            print(f"Model {model_name} is already loaded")
        
        # Call Ollama API for inference
        print(f"Sending request to Ollama with model: {model_name}")
        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                }
            },
            timeout=1800
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Format and return the response
        return {
            "model": model_name,
            "created": int(time.time()),
            "response": result.get("message", {}).get("content", ""),
            "done": True
        }
    except Exception as e:
        # Comprehensive error handling
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    # Update the argument parser to include max_containers and idle_timeout
    parser = argparse.ArgumentParser(
        description="Deploy a configurable Ollama API server using Modal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gpu-type", 
                        choices=["H100", "A100", "L40", "A10G"],
                        help="GPU type to use")
    parser.add_argument("--gpu-count", 
                        type=int, 
                        help="Number of GPUs to use")
    parser.add_argument("--model", 
                        help="Default model name (e.g., gemma3:27b, llama3:8b)")
    parser.add_argument("--max-containers", 
                        type=int,
                        help=f"Maximum number of containers (default: {DEFAULT_MAX_CONTAINERS})")
    parser.add_argument("--idle-timeout", 
                        type=int,
                        help=f"Idle timeout in seconds (default: {DEFAULT_IDLE_TIMEOUT} seconds / {DEFAULT_IDLE_TIMEOUT/60} minutes)")
    parser.add_argument("--deploy", 
                        action="store_true", 
                        help="Deploy the app immediately")
    args = parser.parse_args()
    
    # Set environment variables based on command-line arguments
    if args.gpu_type:
        os.environ["GPU_TYPE"] = args.gpu_type
    if args.gpu_count:
        os.environ["GPU_COUNT"] = str(args.gpu_count)
    if args.model:
        os.environ["MODEL_NAME"] = args.model
    if args.max_containers:
        os.environ["MAX_CONTAINERS"] = str(args.max_containers)
    if args.idle_timeout:
        os.environ["IDLE_TIMEOUT"] = str(args.idle_timeout)
    
    # Display current configuration
    print(f"Configuration:")
    print(f"  GPU Type: {os.environ.get('GPU_TYPE', DEFAULT_GPU_TYPE)}")
    print(f"  GPU Count: {os.environ.get('GPU_COUNT', DEFAULT_GPU_COUNT)}")
    print(f"  Model: {os.environ.get('MODEL_NAME', 'gemma3:27b')}")
    print(f"  Max Containers: {os.environ.get('MAX_CONTAINERS', DEFAULT_MAX_CONTAINERS)}")
    print(f"  Idle Timeout: {os.environ.get('IDLE_TIMEOUT', DEFAULT_IDLE_TIMEOUT)} seconds ({int(os.environ.get('IDLE_TIMEOUT', DEFAULT_IDLE_TIMEOUT))/60:.1f} minutes)")
    
    # Deploy automatically if requested
    if args.deploy:
        print("\nDeploying application...")
        import subprocess
        subprocess.run(["modal", "deploy", __file__])
    else:
        print("\nYou can now deploy this with: modal deploy main.py")
        print("Or with custom configuration: python main.py --gpu-type H100 --model llama3:8b --max-containers 1 --idle-timeout 600 --deploy")