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
import threading
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
import sys
import json

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

# Update default GPU configuration
DEFAULT_GPU_TYPE = "H100"    # Default to H100 for best performance
DEFAULT_GPU_COUNT = 1        # Use single H100 for quantized model

# Add quantization settings
QUANTIZATION_LEVELS = {
    "q4_0": "4-bit quantization (fastest, lowest quality)",
    "q4_1": "4-bit quantization (balanced)",
    "q5_0": "5-bit quantization (balanced)",
    "q5_1": "5-bit quantization (better quality)",
    "q8_0": "8-bit quantization (best quality, slower)"
}

# Add this with the other default configurations
DEFAULT_MAX_CONTAINERS = 1  # Default to 1 container since we're using 2 GPUs

# Add this near the top of the file with other constants
OLLAMA_DATA_DIR = "/ollama_data"  # New mount point for our volume
MODEL_STATE_DIR = f"{OLLAMA_DATA_DIR}/model_states"
MODEL_CACHE_DIR = f"{OLLAMA_DATA_DIR}/model_cache"
MODEL_STATE_TIMEOUT = 3600  # 1 hour timeout for cached states
MODEL_LOAD_TIMEOUT = 300  # 5 minutes timeout for model loading
WARM_START_FILE = "/root/.ollama/warm_start.txt"

# Add this near the other constants
FIRST_DEPLOY_FLAG = f"{OLLAMA_DATA_DIR}/.first_deploy_done"

# Add these global variables at the top of the file after imports
loaded_models = set()
model_gpu_states = {}  # Track GPU memory state for each model
ollama_process = None  # Initialize ollama_process as None

def get_gpu_config():
    """
    Get GPU configuration from environment variables or use defaults.
    """
    gpu_type = os.environ.get("GPU_TYPE", DEFAULT_GPU_TYPE)
    gpu_count_str = os.environ.get("GPU_COUNT", str(DEFAULT_GPU_COUNT))
    
    try:
        gpu_count = int(gpu_count_str)
    except ValueError:
        print(f"⚠️ Warning: Invalid GPU count '{gpu_count_str}', using default: {DEFAULT_GPU_COUNT}")
        gpu_count = DEFAULT_GPU_COUNT
    
    # Validate GPU type against supported options
    valid_gpu_types = ["H100", "A100", "L40", "A10G"]
    
    if gpu_type not in valid_gpu_types:
        print(f"⚠️ Warning: Unknown GPU type '{gpu_type}', falling back to {DEFAULT_GPU_TYPE}")
        gpu_type = DEFAULT_GPU_TYPE
    
    # Format the GPU specification string
    gpu_spec = gpu_type
    if gpu_count > 1:
        gpu_spec += f":{gpu_count}"
    
    print(f"🖥️ Using GPU configuration: {gpu_spec}")
    
    # Only verify GPU in container environment
    if os.environ.get("MODAL_ENVIRONMENT") == "container":
        try:
            # First check if nvidia-smi is available
            try:
                nvidia_smi_output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
                print("✓ NVIDIA-SMI available")
                print(nvidia_smi_output)
            except subprocess.CalledProcessError as e:
                print(f"⚠️ NVIDIA-SMI error: {e.output.decode()}")
                raise Exception("NVIDIA-SMI not available")
            except FileNotFoundError:
                print("⚠️ NVIDIA-SMI not found")
                raise Exception("NVIDIA-SMI not installed")
            
            # Then check CUDA availability
            try:
                nvcc_output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
                print("✓ CUDA compiler available")
                print(nvcc_output)
            except subprocess.CalledProcessError as e:
                print(f"⚠️ NVCC error: {e.output.decode()}")
                raise Exception("CUDA compiler not available")
            except FileNotFoundError:
                print("⚠️ NVCC not found")
                raise Exception("CUDA compiler not installed")
            
            # Finally check PyTorch CUDA support
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                print(f"✓ CUDA available in PyTorch: {torch.cuda.get_device_capability(0)}")
                print(f"✓ GPU detected: {device_name}")
                print(f"✓ Number of GPUs: {device_count}")
                
                # Verify GPU type matches requested type
                if gpu_type.lower() not in device_name.lower():
                    print(f"⚠️ Warning: Requested {gpu_type} but detected {device_name}")
            else:
                print("⚠️ Warning: PyTorch CUDA not available")
                raise Exception("PyTorch CUDA not available")
                
        except ImportError:
            print("⚠️ Warning: PyTorch not available")
            raise Exception("PyTorch not installed")
        except Exception as e:
            print(f"⚠️ Warning: GPU verification failed: {str(e)}")
            raise
    else:
        print("ℹ️ Local environment detected, skipping GPU verification")
    
    return gpu_spec

# Create persistent volume for storing Ollama models and states between container restarts
model_cache = modal.Volume.from_name("ollama-model-cache", create_if_missing=True)

# Base image configuration with required dependencies
base_image = (
    modal.Image.from_registry("python:3.10-slim")
    .run_commands([
        # Install CUDA and NVIDIA dependencies
        "apt-get update",
        "apt-get install -y curl pciutils lshw gnupg2 software-properties-common",
        "curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o cuda-keyring.deb",
        "dpkg -i cuda-keyring.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-4",
        # Install Ollama
        "curl -fsSL https://ollama.ai/install.sh | sh",
        "chmod +x /usr/local/bin/ollama",
        # Create initial Ollama directory
        "mkdir -p /root/.ollama",
        # Verify installations
        "lspci | grep -i nvidia || echo 'No NVIDIA GPU found'",
        "lshw -C display || echo 'No display devices found'",
        "nvcc --version || echo 'CUDA toolkit not found'",
        "python --version || echo 'Python not found'",
        "which python || echo 'Python not in PATH'",
        "which pip || echo 'Pip not in PATH'"
    ])
    .pip_install(
        "requests",
        "psutil",
        "fastapi[standard]",
        "uvicorn",
        "torch",
        "numpy",
        "nvidia-ml-py3"  # For better GPU monitoring
    )
    .env({
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin",
        "CUDA_VISIBLE_DEVICES": "0",  # Use single GPU
        "OLLAMA_GPU_LAYERS": "-1",     # Use all GPU layers
        "OLLAMA_NUM_THREAD": "1",      # Minimize CPU threads
        "OLLAMA_BATCH_SIZE": "8192",   # Increased batch size for faster loading
        "OLLAMA_NUM_CTX": "16384",     # Increased context window
        "OLLAMA_NUM_BATCH": "8192",    # Increased batch size
        "OLLAMA_NUM_GPU": "1",         # Use single GPU
        "OLLAMA_QUANTIZATION": "q4_0", # Use 4-bit quantization
        # OpenMP settings
        "OMP_NUM_THREADS": "1",        # Limit OpenMP threads
        "OMP_SCHEDULE": "static",      # Static scheduling
        "OMP_PROC_BIND": "close",      # Bind threads close
        "OMP_PLACES": "cores",         # Place threads on cores
        "OMP_DISPLAY_ENV": "true",     # Show OpenMP environment
        "OMP_WAIT_POLICY": "active",   # Active wait policy
        "OMP_DYNAMIC": "false",        # Disable dynamic adjustment
        "OMP_NESTED": "false",         # Disable nested parallelism
        "OMP_MAX_ACTIVE_LEVELS": "1",  # Limit active levels
        # Other thread settings
        "MKL_NUM_THREADS": "1",        # Limit MKL threads
        "NUMEXPR_NUM_THREADS": "1",    # Limit NumExpr threads
        # Force GPU memory allocation
        "OLLAMA_GPU_MEMORY": "70",     # Request 70GB of GPU memory (for quantized model)
        "OLLAMA_GPU_LAYERS": "-1",     # Use all GPU layers
        "OLLAMA_NUM_GPU": "1",         # Use single GPU
        "OLLAMA_BATCH_SIZE": "8192",   # Increased batch size
        "OLLAMA_NUM_BATCH": "8192",    # Increased batch size
        "OLLAMA_NUM_CTX": "16384",     # Increased context window
        # Additional CUDA settings
        "CUDA_LAUNCH_BLOCKING": "1",   # Synchronous CUDA operations
        "CUDA_CACHE_DISABLE": "0",     # Enable CUDA cache
        "CUDA_CACHE_PATH": "/tmp/cuda-cache",  # Set cache path
        "CUDA_CACHE_MAXSIZE": "4294967296",  # 4GB cache size
        # NVIDIA specific settings
        "NVIDIA_TF32_OVERRIDE": "1",   # Enable TF32
        "NVIDIA_GPU_MEMORY_FRACTION": "0.95",  # Use 95% of GPU memory
        "NVIDIA_GPU_MEMORY_ALLOCATION": "0.95",  # Allocate 95% of GPU memory
        # Force memory allocation
        "CUDA_MEMORY_ALLOCATION": "0.95",  # Allocate 95% of GPU memory
        "CUDA_MEMORY_FRACTION": "0.95",    # Use 95% of GPU memory
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",  # Limit device connections
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",  # Order devices by PCI bus ID
        "CUDA_VISIBLE_DEVICES": "0",      # Use single GPU
        # PyTorch specific settings
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Limit split size
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8",  # Aggressive GC
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # Allow expansion
        "PYTORCH_CUDA_ALLOC_CONF": "roundup_power2_divisions:4"  # Round up allocations
    })
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

# Update the verify_gpu_memory function
def verify_gpu_memory():
    """Verify GPU memory allocation with improved efficiency."""
    try:
        # Import required modules inside the function
        import torch
        import numpy as np
        import pynvml
        
        # Initialize NVML
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Check GPU memory before loading
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_used = info.used // 1024 // 1024  # Convert to MB
        gpu_total = info.total // 1024 // 1024
        print(f"Initial GPU Memory: {gpu_used}MB/{gpu_total}MB ({gpu_used/gpu_total*100:.1f}%)")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available")
        
        # Force GPU memory allocation with a single large tensor
        print("Forcing GPU memory allocation...")
        try:
            # Allocate a single large tensor (4GB)
            tensor = torch.zeros((1024, 1024, 1024), device='cuda')
            torch.cuda.synchronize()
            
            # Check memory after allocation
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = info.used // 1024 // 1024
            print(f"GPU Memory after allocation: {gpu_used}MB/{gpu_total}MB ({gpu_used/gpu_total*100:.1f}%)")
            
            # Keep the tensor allocated
            print("Keeping tensor allocated for model loading...")
            return True
            
        except Exception as e:
            print(f"Warning: Could not allocate GPU tensor: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Failed to verify GPU memory: {str(e)}")
        return False

# Update the monitor_memory function
def monitor_memory():
    """Monitor GPU memory usage with reduced warning messages."""
    try:
        # Import required modules inside the function
        import pynvml
        import torch
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"Failed to initialize NVML: {str(e)}")
        return
        
    last_warning_time = 0
    warning_cooldown = 30  # Only show warnings every 30 seconds
    last_memory_usage = 0
    memory_threshold = 0.1  # 10% change threshold
    
    while True:
        try:
            # Check GPU memory using NVML
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used = info.used // 1024 // 1024
            gpu_total = info.total // 1024 // 1024
            gpu_percent = (gpu_used / gpu_total) * 100
            
            # Get PyTorch memory stats
            torch_memory = torch.cuda.memory_allocated() // 1024 // 1024
            torch_cached = torch.cuda.memory_reserved() // 1024 // 1024
            
            # Only print if memory usage changes significantly
            current_time = time.time()
            memory_change = abs(gpu_used - last_memory_usage) / gpu_total
            
            if current_time - last_warning_time >= warning_cooldown and memory_change > memory_threshold:
                if gpu_percent < 20:  # Less than 20% usage
                    print(f"\nℹ️ GPU Memory: {gpu_used}MB/{gpu_total}MB ({gpu_percent:.1f}%)", flush=True)
                    last_warning_time = current_time
                elif gpu_percent > 90:  # High memory usage
                    print(f"\n⚠️ High GPU memory usage: {gpu_percent:.1f}%", flush=True)
                    last_warning_time = current_time
                
                last_memory_usage = gpu_used
            
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            print(f"Memory monitoring error: {str(e)}")
            time.sleep(5)
            continue

def wait_for_model_ready(model_name, max_attempts=6, initial_wait=10):
    """Wait for model to be ready with progressive wait periods."""
    for attempt in range(max_attempts):
        try:
            # Progressive wait time (10s, 20s, 30s, 40s, 50s, 60s)
            wait_time = initial_wait * (attempt + 1)
            print(f"Waiting for model to be ready (attempt {attempt + 1}/{max_attempts}, {wait_time}s)...", flush=True)
            time.sleep(wait_time)
            
            # Try a minimal inference
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "test",
                    "stream": False,
                    "options": {
                        "num_ctx": 64,      # Minimal context
                        "num_batch": 64,    # Minimal batch
                        "batch_size": 64,   # Minimal batch
                        "num_gpu": 1,
                        "num_thread": 1,
                        "gpu_layers": -1
                    }
                },
                timeout=wait_time
            )
            
            if response.status_code == 200:
                print(f"✓ Model {model_name} is ready after {wait_time}s", flush=True)
                return True
                
        except requests.exceptions.Timeout:
            print(f"⚠️ Model not ready after {wait_time}s, waiting longer...", flush=True)
            continue
        except Exception as e:
            print(f"⚠️ Error checking model readiness: {str(e)}", flush=True)
            continue
            
    print("⚠️ Model failed to become ready after all attempts", flush=True)
    return False

def force_model_load(model_name):
    """Force model into GPU memory with improved error handling and persistence."""
    try:
        print(f"\nForcing model {model_name} into GPU memory...")
        
        # First check if model is already loaded
        if is_model_loaded_in_gpu(model_name):
            print("✓ Model already loaded in GPU memory")
            return True
            
        # Force model load with optimized settings
        print("Starting model load with optimized settings...")
        
        # Allocate GPU memory more efficiently
        import torch
        try:
            # Allocate a single large tensor to reserve GPU memory
            tensor = torch.zeros((1024, 1024, 1024), device='cuda')  # 4GB tensor
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Could not allocate GPU tensor: {str(e)}")
        
        # Try direct model load with optimized settings
        print("Attempting direct model load with optimized settings...")
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "test",
                    "stream": False,
                    "options": {
                        "num_gpu": 1,        # Single GPU
                        "num_thread": 1,     # Minimize CPU threads
                        "gpu_layers": -1,    # Use all GPU layers
                        "batch_size": 512,   # Optimized batch size
                        "num_ctx": 512,      # Optimized context window
                        "num_batch": 512,    # Optimized batch size
                        "quantization": "q4_0",  # Use 4-bit quantization
                        "num_gqa": 1,        # Group query attention
                        "num_keep": 0,       # Don't keep tokens
                        "seed": -1,          # Random seed
                        "tfs_z": 1,          # Tail free sampling
                        "typical_p": 1,      # Typical sampling
                        "repeat_last_n": 64, # Repeat context
                        "repeat_penalty": 1.1,# Repeat penalty
                        "presence_penalty": 0,# No presence penalty
                        "frequency_penalty": 0,# No frequency penalty
                        "mirostat": 0,       # Disable mirostat
                        "mirostat_tau": 5,   # Default mirostat tau
                        "mirostat_eta": 0.1, # Default mirostat eta
                        "rope_scaling": None # Disable rope scaling
                    }
                },
                timeout=180  # 3 minute timeout
            )
            
            if response.status_code == 200:
                print("✓ Model loaded successfully with optimized settings")
                # Save the successful state
                model_gpu_states[model_name] = {
                    "num_gpu": 1,
                    "num_thread": 1,
                    "gpu_layers": -1,
                    "batch_size": 512,
                    "num_ctx": 512,
                    "num_batch": 512,
                    "quantization": "q4_0"
                }
                loaded_models.add(model_name)
                
                # Wait for model to be ready with progressive wait
                if wait_for_model_ready(model_name):
                    print("✓ Model is fully ready for inference", flush=True)
                    return True
                else:
                    print("⚠️ Model loaded but failed to become ready", flush=True)
                    return False
                
            else:
                print(f"⚠️ Direct load failed with status {response.status_code}")
        except requests.exceptions.Timeout:
            print("⚠️ Direct load timed out, trying fallback method")
        except Exception as e:
            print(f"⚠️ Direct load failed: {str(e)}")
        
        # If direct load fails, try progressive loading with shorter timeouts
        print("Starting progressive model load...")
        batch_sizes = [256, 512]  # Reduced number of attempts
        
        for batch_size in batch_sizes:
            try:
                print(f"Trying batch size: {batch_size}")
                model_options = {
                    "num_gpu": 1,           # Single GPU
                    "num_thread": 1,        # Minimize CPU threads
                    "gpu_layers": -1,       # Use all GPU layers
                    "batch_size": batch_size,
                    "num_ctx": batch_size,
                    "num_batch": batch_size,
                    "quantization": "q4_0",  # Use 4-bit quantization
                    "num_gqa": 1,           # Group query attention
                    "num_keep": 0,          # Don't keep tokens
                    "seed": -1,             # Random seed
                    "tfs_z": 1,             # Tail free sampling
                    "typical_p": 1,         # Typical sampling
                    "repeat_last_n": 64,    # Repeat context
                    "repeat_penalty": 1.1,   # Repeat penalty
                    "presence_penalty": 0,   # No presence penalty
                    "frequency_penalty": 0,  # No frequency penalty
                    "mirostat": 0,          # Disable mirostat
                    "mirostat_tau": 5,      # Default mirostat tau
                    "mirostat_eta": 0.1,    # Default mirostat eta
                    "rope_scaling": None    # Disable rope scaling
                }
                
                # Try to load with current batch size
                response = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "test",
                        "stream": False,
                        "options": model_options
                    },
                    timeout=180  # 3 minute timeout
                )
                
                if response.status_code == 200:
                    print(f"✓ Model loaded successfully with batch size {batch_size}")
                    # Save the successful state
                    model_gpu_states[model_name] = model_options
                    loaded_models.add(model_name)
                    
                    # Wait for model to be ready with progressive wait
                    if wait_for_model_ready(model_name):
                        print("✓ Model is fully ready for inference", flush=True)
                        return True
                    else:
                        print("⚠️ Model loaded but failed to become ready", flush=True)
                        return False
                        
                else:
                    print(f"⚠️ Model load returned status {response.status_code} with batch size {batch_size}")
                    time.sleep(2)  # Short delay between attempts
            except requests.exceptions.Timeout:
                print(f"⚠️ Model load timed out with batch size {batch_size}, trying next size")
                time.sleep(2)
                continue
            except Exception as e:
                print(f"⚠️ Error with batch size {batch_size}: {str(e)}")
                time.sleep(2)
                continue
        
        print("⚠️ All loading attempts failed")
        return False
            
    except requests.exceptions.Timeout:
        print("⚠️ Model load timed out, but may still be usable")
        return True
    except Exception as e:
        print(f"⚠️ Error forcing model load: {str(e)}")
        return False
    finally:
        # Clean up tensors
        try:
            del tensor
            torch.cuda.empty_cache()
        except:
            pass

# Add this function to check Ollama service health
def check_ollama_health():
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

# Add this function to start Ollama service
def start_ollama_service():
    """Start the Ollama service with proper error handling and verification."""
    try:
        # Kill any existing Ollama processes
        try:
            subprocess.run(["pkill", "-f", "ollama"], stderr=subprocess.DEVNULL)
            time.sleep(2)  # Give time for process to fully terminate
        except:
            pass

        print("Starting Ollama service...")
        
        # Start Ollama with proper environment variables
        env = os.environ.copy()
        env["OLLAMA_HOST"] = "127.0.0.1"
        env["OLLAMA_ORIGINS"] = "*"
        env["OLLAMA_MODELS"] = "/root/.ollama/models"
        
        # Start Ollama in background with proper logging
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=open("/tmp/ollama_stdout.log", "w"),
            stderr=open("/tmp/ollama_stderr.log", "w"),
            env=env
        )
        
        # Wait for service to start with better error handling
        max_attempts = 12
        for attempt in range(max_attempts):
            print(f"Waiting for Ollama service to start (attempt {attempt + 1}/{max_attempts})...")
            try:
                # Check if process is still running
                if process.poll() is not None:
                    print("⚠️ Ollama process died unexpectedly")
                    # Read error logs
                    with open("/tmp/ollama_stderr.log", "r") as f:
                        print(f"Error logs: {f.read()}")
                    raise Exception("Ollama process died")
                
                # Try to connect to service
                response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✓ Ollama service is ready")
                    # Additional verification
                    time.sleep(2)  # Give extra time for full initialization
                    try:
                        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
                        if response.status_code == 200:
                            print("✓ Ollama service fully initialized")
                            return True
                    except:
                        pass
            except requests.exceptions.RequestException:
                time.sleep(5)  # Wait longer between attempts
                continue
            except Exception as e:
                print(f"⚠️ Error checking Ollama service: {str(e)}")
                time.sleep(5)
                continue
        
        print("⚠️ Failed to start Ollama service after all attempts")
        return False
        
    except Exception as e:
        print(f"⚠️ Error starting Ollama service: {str(e)}")
        return False

# Add function to manage model state
def save_model_state(model_name, state_data):
    """Save model state to persistent storage"""
    try:
        os.makedirs(MODEL_STATE_DIR, exist_ok=True)
        state_file = os.path.join(MODEL_STATE_DIR, f"{model_name}.state")
        with open(state_file, "w") as f:
            f.write(json.dumps({
                "model": model_name,
                "timestamp": int(time.time()),
                "state": state_data
            }))
        print(f"✓ Model state saved for {model_name}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to save model state: {str(e)}")
        return False

def load_model_state(model_name):
    """Load model state from persistent storage"""
    try:
        state_file = os.path.join(MODEL_STATE_DIR, f"{model_name}.state")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.loads(f.read())
                # Check if state is still valid (within timeout)
                if time.time() - state["timestamp"] < MODEL_STATE_TIMEOUT:
                    print(f"✓ Found cached state for {model_name} from {datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                    return state["state"]
                else:
                    print(f"⚠️ Cached state for {model_name} expired at {datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        return None
    except Exception as e:
        print(f"⚠️ Failed to load model state: {str(e)}")
        return None

def delete_model_state(model_name=None):
    """Delete cached model state(s) from persistent storage"""
    try:
        if model_name:
            # Delete specific model state
            state_file = os.path.join(MODEL_STATE_DIR, f"{model_name}.state")
            if os.path.exists(state_file):
                os.remove(state_file)
                print(f"✓ Deleted cached state for {model_name}")
            else:
                print(f"ℹ️ No cached state found for {model_name}")
        else:
            # Delete all model states
            if os.path.exists(MODEL_STATE_DIR):
                for file in os.listdir(MODEL_STATE_DIR):
                    if file.endswith('.state'):
                        os.remove(os.path.join(MODEL_STATE_DIR, file))
                print("✓ Deleted all cached model states")
            else:
                print("ℹ️ No cached states directory found")
        return True
    except Exception as e:
        print(f"⚠️ Failed to delete model state: {str(e)}")
        return False

# Add this function near the other cache management functions
def clear_all_caches():
    """Clear both model state cache and Modal volume cache"""
    try:
        # Check if this is first deployment
        if os.path.exists(FIRST_DEPLOY_FLAG):
            print("ℹ️ First deployment already completed, preserving caches")
            return True
            
        print("First deployment detected, clearing all caches...")
        
        # Clear model state cache
        if os.path.exists(MODEL_STATE_DIR):
            for file in os.listdir(MODEL_STATE_DIR):
                if file.endswith('.state'):
                    os.remove(os.path.join(MODEL_STATE_DIR, file))
            print("✓ Cleared model state cache")
        
        # Clear Modal volume cache
        if os.path.exists(OLLAMA_DATA_DIR):
            # Remove all files except .initialized
            for item in os.listdir(OLLAMA_DATA_DIR):
                if item != '.initialized':
                    path = os.path.join(OLLAMA_DATA_DIR, item)
                    if os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
            print("✓ Cleared Modal volume cache")
            
        # Clear Ollama cache
        ollama_cache = "/root/.ollama"
        if os.path.exists(ollama_cache):
            for item in os.listdir(ollama_cache):
                path = os.path.join(ollama_cache, item)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            print("✓ Cleared Ollama cache")
            
        # Mark first deployment as complete
        os.makedirs(os.path.dirname(FIRST_DEPLOY_FLAG), exist_ok=True)
        with open(FIRST_DEPLOY_FLAG, 'w') as f:
            f.write(str(int(time.time())))
        print("✓ Marked first deployment as complete")
            
        return True
    except Exception as e:
        print(f"⚠️ Failed to clear caches: {str(e)}")
        return False

def check_model_exists(model_name):
    """Check if model exists in Ollama service."""
    try:
        response = requests.get(f"http://127.0.0.1:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return any(model.get('name') == model_name or model.get('name').startswith(f"{model_name}:") for model in models)
        return False
    except:
        return False

def sync_files():
    """Sync files to disk only when necessary."""
    try:
        # Only sync if we're in a container environment
        if os.environ.get("MODAL_ENVIRONMENT") == "container":
            subprocess.run(["sync"], check=True)
    except Exception as e:
        print(f"Warning: File sync failed: {str(e)}")

def is_model_loaded_in_gpu(model_name):
    """Check if model is loaded in GPU memory."""
    try:
        # First check if we have a saved GPU state
        if model_name in model_gpu_states:
            # Try a quick inference to verify the model is still loaded
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "test",
                    "stream": False,
                    "options": {
                        "num_ctx": 128,     # Minimal context
                        "num_batch": 128,   # Minimal batch
                        "batch_size": 128   # Minimal batch
                    }
                },
                timeout=5
            )
            if response.status_code == 200:
                print(f"✓ Model {model_name} verified in GPU memory", flush=True)
                return True
            else:
                print(f"⚠️ Model {model_name} not responding in GPU memory", flush=True)
                # Remove from both tracking sets if not responding
                if model_name in model_gpu_states:
                    del model_gpu_states[model_name]
                if model_name in loaded_models:
                    loaded_models.remove(model_name)
                return False
        return False
    except Exception as e:
        print(f"⚠️ Error checking GPU memory state: {str(e)}", flush=True)
        return False

def check_model_status(model_name, timeout=30):
    """Check if model is ready for inference with increased timeout."""
    try:
        # First check if model is in our tracking sets
        if model_name in loaded_models and model_name in model_gpu_states:
            # Verify model is responding with increased timeout
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "test",
                    "stream": False,
                    "options": {
                        "num_ctx": 128,
                        "num_batch": 128,
                        "batch_size": 128,
                        "num_gpu": 1,
                        "num_thread": 1,
                        "gpu_layers": -1
                    }
                },
                timeout=timeout
            )
            if response.status_code == 200:
                print(f"✓ Model {model_name} verified and ready", flush=True)
                return True
            else:
                print(f"⚠️ Model {model_name} not responding", flush=True)
                # Clear tracking state
                if model_name in model_gpu_states:
                    del model_gpu_states[model_name]
                if model_name in loaded_models:
                    loaded_models.remove(model_name)
                return False
        return False
    except requests.exceptions.Timeout:
        print(f"⚠️ Model status check timed out after {timeout} seconds, model may still be loading", flush=True)
        return False
    except Exception as e:
        print(f"⚠️ Error checking model status: {str(e)}", flush=True)
        return False

@app.function(
    image=base_image,
    gpu=get_gpu_config(),
    volumes={OLLAMA_DATA_DIR: model_cache},
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
    global ollama_process, loaded_models, model_gpu_states
    
    try:
        # Extract data from the request
        prompt = request_data.get("prompt", "Hello")
        temperature = request_data.get("temperature", 0.7)
        model_name = request_data.get("model", MODEL_NAME)
        
        print(f"Received request with prompt: {prompt}")
        print(f"Using model: {model_name}")
        print(f"Currently loaded models: {loaded_models}", flush=True)
        
        # Start Ollama service if not already running
        if ollama_process is None or not check_ollama_health():
            print("Starting Ollama service...")
            try:
                # Set up data directory structure after volume is mounted
                subprocess.run([
                    "bash", "-c", f"""
                    # Create directories in mounted volume
                    mkdir -p {OLLAMA_DATA_DIR}
                    mkdir -p {MODEL_STATE_DIR}
                    
                    # Copy initial Ollama data if volume is empty
                    if [ ! -f {OLLAMA_DATA_DIR}/.initialized ]; then
                        cp -r /root/.ollama/* {OLLAMA_DATA_DIR}/ || true
                        touch {OLLAMA_DATA_DIR}/.initialized
                    fi
                    
                    # Create symlink to maintain compatibility
                    rm -rf /root/.ollama
                    ln -s {OLLAMA_DATA_DIR} /root/.ollama
                    """
                ], check=True)
                
                # Start Ollama service
                ollama_process = start_ollama_service()
                
            except Exception as e:
                print(f"Error starting Ollama service: {str(e)}")
                raise
        
        # Ensure requested model is loaded
        if model_name not in loaded_models or not check_model_status(model_name):
            print(f"Checking model {model_name}...", flush=True)
            try:
                # First verify Ollama is ready and fully initialized
                for _ in range(3):  # Try up to 3 times
                    try:
                        health_check = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
                        if health_check.status_code == 200:
                            print("✓ Ollama service verified", flush=True)
                            break
                    except Exception as e:
                        print(f"⚠️ Service check failed: {str(e)}", flush=True)
                        time.sleep(2)
                else:
                    raise Exception("Ollama service not ready")

                # Check if model exists in Ollama
                if check_model_exists(model_name):
                    print(f"✓ Model {model_name} found in Ollama", flush=True)
                else:
                    print(f"Pulling model {model_name}...", flush=True)
                    # Temporarily disable GPU for pulling
                    original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    
                    try:
                        # Use subprocess.run with timeout
                        result = subprocess.run(
                            ["ollama", "pull", model_name],
                            capture_output=True,
                            text=True,
                            timeout=1800,  # 30 minute timeout
                            env=os.environ.copy()
                        )
                        
                        # Print output in real-time
                        if result.stdout:
                            print(result.stdout, flush=True)
                        if result.stderr:
                            print(result.stderr, flush=True)
                        
                        if result.returncode != 0:
                            raise Exception(f"Model pull failed with return code {result.returncode}")
                        
                        # Verify model was pulled successfully
                        if check_model_exists(model_name):
                            print(f"✓ Model {model_name} pulled successfully", flush=True)
                            # Only sync once after successful pull
                            sync_files()
                        else:
                            raise Exception(f"Model {model_name} not found after pull")
                            
                    finally:
                        # Restore GPU access
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}", flush=True)
                raise

            # Load model into GPU memory
            print("Loading model into GPU memory...", flush=True)
            try:
                # Verify GPU memory allocation
                if not verify_gpu_memory():
                    raise Exception("Failed to allocate GPU memory")
                
                print("Performing model load...", flush=True)
                
                # Start memory monitoring thread
                memory_thread = threading.Thread(target=monitor_memory, daemon=True)
                memory_thread.start()
                
                try:
                    # Force model load with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        print(f"\nAttempt {attempt + 1}/{max_retries} to load model...", flush=True)
                        if force_model_load(model_name):
                            print("✓ Model loaded successfully", flush=True)
                            # Verify model is ready
                            if check_model_status(model_name):
                                print("✓ Model verified and ready for inference", flush=True)
                                break
                            else:
                                print("⚠️ Model loaded but not responding, retrying...", flush=True)
                                time.sleep(5)
                        elif attempt < max_retries - 1:
                            print("⚠️ Model load failed, retrying...", flush=True)
                            time.sleep(5)
                        else:
                            print("⚠️ Model load failed after all attempts", flush=True)
                    
                    # Give it a moment to stabilize
                    time.sleep(5)
                    
                except requests.exceptions.Timeout:
                    print("\n⚠️ Preload timed out, but model may still be usable. Continuing...", flush=True)
                    time.sleep(5)
                except Exception as e:
                    print(f"\n⚠️ Preload failed: {str(e)}, but model may still be usable. Continuing...", flush=True)
                    time.sleep(5)
                finally:
                    # Stop the memory monitoring thread
                    memory_thread.join(timeout=1)
            except Exception as e:
                print(f"Error loading model into GPU: {str(e)}", flush=True)
                raise
        else:
            print(f"Model {model_name} is already loaded and ready", flush=True)
        
        # Call Ollama API for inference
        print(f"Sending request to Ollama with model: {model_name}")
        try:
            # Verify model is ready one last time
            if not check_model_status(model_name):
                raise Exception("Model not ready for inference")
            
            print("Starting inference...", flush=True)
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
                        "num_gpu": 1,        # Use single GPU
                        "num_thread": 1,     # Minimize CPU threads
                        "batch_size": 512,   # Optimize batch size
                        "gpu_layers": -1,    # Use all GPU layers
                        "num_ctx": 4096,     # Context window
                        "num_batch": 512,    # Batch size
                        "num_gqa": 1,        # Group query attention
                        "num_keep": 0,       # Don't keep tokens
                        "seed": -1,          # Random seed
                        "tfs_z": 1,          # Tail free sampling
                        "typical_p": 1,      # Typical sampling
                        "repeat_last_n": 64, # Repeat context
                        "repeat_penalty": 1.1,# Repeat penalty
                        "presence_penalty": 0,# No presence penalty
                        "frequency_penalty": 0,# No frequency penalty
                        "mirostat": 0,       # Disable mirostat
                        "mirostat_tau": 5,   # Default mirostat tau
                        "mirostat_eta": 0.1, # Default mirostat eta
                        "rope_scaling": None # Disable rope scaling
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
        except requests.exceptions.Timeout:
            print("Inference request timed out after 30 minutes", flush=True)
            raise Exception("Inference request timed out")
        except Exception as e:
            print(f"Error during inference: {str(e)}", flush=True)
            raise
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
    parser.add_argument("--clear-cache",
                        action="store_true",
                        help="Clear all cached model states")
    parser.add_argument("--clear-model-cache",
                        help="Clear cached state for specific model")
    parser.add_argument("--clear-all-caches",
                        action="store_true",
                        help="Clear all caches including Modal volume and Ollama cache")
    parser.add_argument("--reset-first-deploy",
                        action="store_true",
                        help="Reset first deployment flag to clear caches again")
    args = parser.parse_args()
    
    # Handle cache deletion first
    if args.reset_first_deploy:
        if os.path.exists(FIRST_DEPLOY_FLAG):
            os.remove(FIRST_DEPLOY_FLAG)
            print("✓ Reset first deployment flag")
        sys.exit(0)
    elif args.clear_all_caches:
        clear_all_caches()
        sys.exit(0)
    elif args.clear_cache:
        delete_model_state()
        sys.exit(0)
    elif args.clear_model_cache:
        delete_model_state(args.clear_model_cache)
        sys.exit(0)
    
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
    
    # Force GPU type if specified
    if args.gpu_type:
        os.environ["GPU_TYPE"] = args.gpu_type
        print(f"Setting GPU type to: {args.gpu_type}")
    
    # Display current configuration
    print(f"\nConfiguration:")
    print(f"  GPU Type: {os.environ.get('GPU_TYPE', DEFAULT_GPU_TYPE)}")
    print(f"  GPU Count: {os.environ.get('GPU_COUNT', str(DEFAULT_GPU_COUNT))}")
    print(f"  Model: {os.environ.get('MODEL_NAME', 'gemma3:27b')}")
    print(f"  Max Containers: {os.environ.get('MAX_CONTAINERS', str(DEFAULT_MAX_CONTAINERS))}")
    print(f"  Idle Timeout: {os.environ.get('IDLE_TIMEOUT', str(DEFAULT_IDLE_TIMEOUT))} seconds ({int(os.environ.get('IDLE_TIMEOUT', DEFAULT_IDLE_TIMEOUT))/60:.1f} minutes)")
    
    # Deploy automatically if requested
    if args.deploy:
        print("\nDeploying application...")
        # Get GPU configuration after environment variables are set
        gpu_config = get_gpu_config()
        print(f"Using GPU configuration: {gpu_config}")
        
        import subprocess
        subprocess.run(["modal", "deploy", __file__])
    else:
        print("\nYou can now deploy this with: modal deploy main.py")
        print("Or with custom configuration: python main.py --gpu-type H100 --model llama3:8b --max-containers 1 --idle-timeout 600 --deploy")