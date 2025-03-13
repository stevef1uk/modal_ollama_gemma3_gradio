# Gradio GUI for LLaMA.cpp Server
# This script deploys a Gradio GUI on Modal that interacts with a remote LLaMA.cpp server.
# Author: Steven Fisher (stevef@gmail.com)
# 
# Description: This script sets up a Gradio interface on Modal, which communicates
#              with a separate llama.cpp server. It requires the following secrets:
#              - MODAL_SECRET_LLAMA_CPP_API_KEY: API key for accessing the LLaMA.cpp server.
#              - gradio_app_access_key: Access key for securing the Gradio app.
#              - llama_server_url: URL of the remote LLaMA.cpp server.

import modal
import pathlib
import shlex
import subprocess
import os

GRADIO_PORT = 8000

app = modal.App("gradio-app")

# Define the secrets with exact names
llm_secret = modal.Secret.from_name("MODAL_SECRET_LLAMA_CPP_API_KEY")  # This creates env var MODAL_SECRET_LLAMA_CPP_API_KEY_LLAMAKEY
gradio_access_secret = modal.Secret.from_name("gradio_app_access_key")
server_url_secret = modal.Secret.from_name("llama_server_url")

# Create the base image
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("gradio", "fastapi", "uvicorn"))

# Add the GUI script to the image
fname = "gui_for_llm.py"
gradio_script_local_path = pathlib.Path(__file__).parent / fname

if not gradio_script_local_path.exists():
    raise RuntimeError(f"{fname} not found! Place the script with your gradio app in the same directory.")

# Add the local file to the image instead of using Mount
image = image.add_local_file(str(gradio_script_local_path), "/root/gui_for_llm.py")

@app.function(
    image=image,
    secrets=[llm_secret, gradio_access_secret, server_url_secret],
    allow_concurrent_inputs=100,
    concurrency_limit=1,  # Drives number of containers!
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

