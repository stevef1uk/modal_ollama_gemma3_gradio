"""
Gradio GUI for Ollama Gemma 3 27B Server

This script deploys a Gradio GUI on Modal that interacts with a remote Ollama server.
Author: Steven Fisher (stevef@gmail.com)

Description: This script sets up a Gradio interface on Modal, which communicates
             with a separate Ollama server. It requires the following secrets:
             - MODAL_SECRET_LLAMA_CPP_API_KEY: API key for accessing the Ollama server.
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
modal_app = modal.App("gemma-gradio-interface")

# Create Modal secret reference for server URL
server_url_secret = modal.Secret.from_name("llama_server_url")

# Create a persistent FastAPI app
web_app = FastAPI()

def chat_with_llm(access_key: str, message: str):
    """Chat function that uses the LLM service."""
    print(f"Received chat request with message: {message}")
    
    # Retrieve the expected access key from environment variables
    expected_access_key = os.environ.get("MODAL_SECRET_GRADIO_APP_ACCESS_KEY")
    if access_key != expected_access_key:
        yield "Error: Invalid access key."
        return

    # Access the LLM API key and server URL
    api_key = os.environ.get("llamakey")
    server_url = os.environ.get("LLAMA_SERVER_URL")
    
    if not api_key:
        yield "Error: LLM API key not found."
        return
    if not server_url:
        yield "Error: Server URL not found in environment variables"
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    enhanced_prompt = f"Please provide a clear, concise answer to this question: {message}"
    
    payload = {
        "prompt": enhanced_prompt,
        "temperature": 0.7,
        "stream": False
    }

    try:
        print("Making request to server...")
        print(f"Using server URL: {server_url}")
        response = requests.post(
            f"{server_url}/v1/completions",
            headers=headers,
            json=payload,
            timeout=300
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
        gr.Textbox(label="Enter your message", lines=4)
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
    title="Gemma 3 27B Chat Interface",
    description="Enter the access key and your message to chat with the Gemma 3 27B model.",
    live=False,
    flagging_mode="never"
)

# Mount Gradio app to FastAPI
app = mount_gradio_app(
    app=web_app,
    blocks=demo,
    path="/"
)

@modal_app.function(
    secrets=[server_url_secret]
)
@modal.asgi_app()
def serve():
    """Return the persistent ASGI app"""
    return web_app

if __name__ == "__main__":
    uvicorn.run(web_app, host="0.0.0.0", port=8000)