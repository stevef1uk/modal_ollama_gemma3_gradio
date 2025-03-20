# Ollama Server with Multiple LLM Support on Modal

This project deploys a high-performance Ollama server on Modal.com with support for multiple language models, including Gemma 3 27B, Llama 3, Phi 3, and more. It also includes a Gradio UI for easy interaction with the models.

## Overview

The system consists of two main components:
1. An Ollama server running on Modal with GPU acceleration
2. A Gradio web interface for interacting with the models

You have to have a Modal account (a nice container as a service system) and then create the necessary secrets. 
Modal: https://modal.com/use-cases/language-models 
Hugging Face: https://huggingface.co

You can sign-up with both sites and do not need to register with a credit card.

## Setup Instructions

### Prerequisites

1. Create a Modal account at [https://modal.com](https://modal.com/use-cases/language-models)
2. Create a Hugging Face account at [https://huggingface.co](https://huggingface.co)
3. Install the Modal CLI and set it up:
   ```
   pip install modal tabulate
   modal setup
   ```

4. Test your Modal setup:
   ```
   modal run simple1.py
   ```
   This should show you the location on the internet of your ISP and where the remote modal code is running.

Then you can deploy the Ollama server with multiple model support, including Gemma 3 27B on a single or multiple GPU. The default is the A10G GPU as this is suitable for currenrt Ollama models, however, it can be configured to be something else and multiple GPUs can also be configured.

Note: The H100 costs $0.001097 / sec (see: https://modal.com/pricing), but the container goes to sleep after 5 minutes of being idle and therefore costs nothing. As the container using Ollama can handle multiple requests at a time, further requests may be queued if the maximum number of containers is reached. This can be configured by adjusting the max_containers parameter.

### Step 1: Create Required Secrets

#### Hugging Face Token

First, create a Hugging Face token:

1. Navigate to [Hugging Face Tokens Page](https://huggingface.co/docs/hub/en/security-tokens)
2. Click "New token"
3. Choose a name for your token (e.g., my-modal-token)
4. Set the role:
   - ✅ Read → If you just need to download models
   - ✅ Write → If you need to upload models
   - ✅ Admin → Full access (not recommended unless necessary)
5. Click "Generate token" and copy it

Create a Modal secret to store your Hugging Face token:

1. **hf-secret**
   - Create a secret named "hf-secret"
   - Use key: HF_TOKEN and store the Hugging Face token in the value

#### Other Required Secrets

Create the following additional secrets in Modal using their web interface:

2. **MODAL_PROXY_TOKEN_ID**
   - Create a new Modal token: ufrom the console: Settings > Proxy Auth Tokens.
   - Copy the token IDs
   - Create a secret named "MODAL_PROXY_TOKEN_ID" with key "token_id" and the token ID as the value

3. **MODAL_PROXY_TOKEN_SECRET**
   - Use the token secret from the previous step
   - Create a secret named "MODAL_PROXY_TOKEN_SECRET" with key "token_secret" and the token secret as the value

4. **llama_server_url**
   - After deploying the Ollama server, you'll get a URL
   - Create a secret named "llama_server_url" with key "LLAMA_SERVER_URL" and the server URL as the value

5. **gradio_app_access_key**
   - Choose a secure password to protect your Gradio interface
   - Create a secret named "gradio_app_access_key" with key "MODAL_SECRET_GRADIO_APP_ACCESS_KEY" and your chosen password as the value

### Step 2: Deploy the Ollama Server

1. Customize the app name in main.py if desired:
   ```python
   app = modal.App(""ollama-api")  
   ```

2. Deploy the server:
   ```
   modal deploy main.py
   ```
   
   You can also deploy with custom configuration:
   ```
   python main.py --gpu-type H100 --model llama3:8b --max-containers 1 --idle-timeout 600 --deploy
   ```

3. Note the deployment URL, which will look like:
   ```
   https://[username]--ollama-api-api.modal.run
   ```

   You can test the deployment with:
   ```bash
   curl -X POST "https://[username]--ollama-api-api.modal.run" \
        -H "Content-Type: application/json" \
        -H "Modal-Key: [your-token-id]" \
        -H "Modal-Secret: [your-token-secret]" \
        -d '{
          "prompt": "Please tell me why the sky is blue",
          "temperature": 0.7,
          "model": "gemma3:27b"
        }'
   ```

### Step 3: Deploy the Gradio Interface

1. Make sure the `gui_for_llm.py` and `deploy_gradio.py` files are in the same directory

2. Deploy the Gradio interface:
   ```
   modal deploy deploy_gradio.py
   ```

3. Note the Gradio interface URL, which will look like:
   ```
   https://[username]--ollama-gradio-interface-web-app.modal.run
   ```

## Usage

1. Open the Gradio interface URL in your browser

2. Enter the access key you set in the `gradio_app_access_key` secret

3. Select a model from the dropdown menu (default is gemma3:27b)

4. Type your prompt and submit

5. The model will process your request and display the response

## Cost Considerations

- The H100 GPU costs $0.001097/second (see [Modal pricing](https://modal.com/pricing))
- The container goes to sleep after the configured idle timeout (default: 5 minutes) and costs nothing while sleeping
- The container wakes up automatically when a request is received
- By default, the system is configured to use a maximum of 2 containers for parallel processing
- You can adjust the maximum number of containers and idle timeout via environment variables or command-line arguments

## Configuration Options

The Ollama server can be configured with the following options:

- **GPU Type**: H100, A100, L40, or A10G (default: A10G)
- **GPU Count**: Number of GPUs per container (default: 1)
- **Model**: Default model to use (default: gemma3:27b)
- **Max Containers**: Maximum number of containers for parallel processing (default: 2)
- **Idle Timeout**: Time in seconds before container shutdown (default: 300 seconds / 5 minutes)

These options can be set via environment variables or command-line arguments when deploying.

## Available Models

The system supports multiple Ollama models, including:
- gemma3:27b
- llama3:8b
- llama3:70b
- phi3:14b
- mistral:7b
- mixtral:8x7b
- codellama:70b
- llama2:70b
- orca2:13b
- vicuna:13b

## Security

- The Ollama API requires Modal proxy authentication
- The Gradio interface is protected by an access key
- Authentication tokens are stored securely as Modal secrets

## Notes

1. The server includes automatic redirection from root (/) to documentation (/docs)
2. GPU costs vary by type and count - monitor usage accordingly
3. For your first request, the model will be downloaded and cached, which may take a while
4. The API will go to sleep after the configured idle timeout (default: 5 minutes)
5. The API will wake up and start serving requests again when a request comes in
6. Container scaling is limited to control costs (default: max 2 containers)
7. As the container using Ollama can handle multiple requests at a time, additional requests may be queued if the maximum number of containers is reached
