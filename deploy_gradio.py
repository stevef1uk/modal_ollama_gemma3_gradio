# Gradio GUI for Ollama Server with Multiple Model Support
# This script deploys a Gradio GUI on Modal that interacts with a remote Ollama server.
# Author: Steven Fisher (stevef@gmail.com)
# 
# Description: This script sets up a Gradio interface on Modal, which communicates
#              with a separate Ollama server. It requires the following secrets:
#              - MODAL_PROXY_TOKEN_ID: Token ID for Modal proxy authentication
#              - MODAL_PROXY_TOKEN_SECRET: Token secret for Modal proxy authentication
#              - gradio_app_access_key: Access key for securing the Gradio app.
#              - llama_server_url: URL of the remote Ollama server.

import modal
import pathlib
import shlex
import subprocess
import os

GRADIO_PORT = 8000

app = modal.App("ollama-gradio-interface")

# Define the secrets with exact names
proxy_token_id = modal.Secret.from_name("MODAL_PROXY_TOKEN_ID")
proxy_token_secret = modal.Secret.from_name("MODAL_PROXY_TOKEN_SECRET")
gradio_access_secret = modal.Secret.from_name("gradio_app_access_key")
server_url_secret = modal.Secret.from_name("llama_server_url")

# Create the base image
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("gradio", "fastapi", "uvicorn", "requests"))

# Add the GUI script to the image
fname = "gui_for_llm.py"
gradio_script_local_path = pathlib.Path(__file__).parent / fname

if not gradio_script_local_path.exists():
    raise RuntimeError(f"{fname} not found! Place the script with your gradio app in the same directory.")

# Add the local file to the image instead of using Mount
image = image.add_local_file(str(gradio_script_local_path), "/root/gui_for_llm.py")

@app.function(
    image=image,
    secrets=[proxy_token_id, proxy_token_secret, gradio_access_secret, server_url_secret],
    allow_concurrent_inputs=100,
    max_containers=1,  # Drives number of containers!
)
@modal.web_server(GRADIO_PORT, startup_timeout=60)
def web_app():
    cmd = f"python /root/gui_for_llm.py --host 0.0.0.0 --port {GRADIO_PORT}"
    subprocess.Popen(cmd, shell=True)

    # Wait for the server to start
    import time
    time.sleep(2)  # Give the server a moment to start

    return "Gradio web server is running"

if __name__ == "__main__":
    modal.runner.deploy_stub(app)
