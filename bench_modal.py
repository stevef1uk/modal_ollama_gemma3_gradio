# This file was evolved from the on eprovided in: https://github.com/MinhNgyuen/llm-benchmark
import argparse
from typing import List
import requests
import json
import os
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

# Set the Modal URL and authentication
MODAL_URL = "https://[your-id]--ollama-api-api.modal.run/"
TOKEN_ID = os.environ.get("TOKEN_ID", "")  # Get from environment variable
TOKEN_SECRET = os.environ.get("TOKEN_SECRET", "")  # Get from environment variable

# Expanded list of potential models to benchmark
DEFAULT_MODELS = [
    "gemma3:27b", 
    "llama3:8b", 
    "phi3:14b",
    "mistral:7b",
    "DeepSeek-R1:14b",
    "llama3.3"
]

class Message(BaseModel):
    role: str
    content: str

class OllamaResponse(BaseModel):
    model: str
    created_at: datetime = Field(default_factory=datetime.now)
    message: Message
    done: bool = True
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print("\nWarning: prompt token count not provided, possibly due to caching.\n")
            return 0  # Set default value
        return value

def estimate_token_count(text):
    """Estimate token count based on words and characters (rough approximation)"""
    if not text:
        return 0
    
    # A more nuanced approximation based on typical tokenization patterns
    # Most tokenizers use roughly 100 tokens per 75 words
    # or about 4 characters per token on average for English text
    
    # Count by words (splitting on whitespace)
    word_count = len(text.split())
    word_based_estimate = word_count * 1.33  # ~4/3 tokens per word
    
    # Count by characters
    char_based_estimate = len(text) / 4  # ~4 chars per token
    
    # Average the two approaches for a better estimate
    return int((word_based_estimate + char_based_estimate) / 2)

def run_benchmark(model_name: str, prompt: str, verbose: bool, timeout: int = 120) -> OllamaResponse:
    headers = {
        "Content-Type": "application/json",
        "Modal-Key": TOKEN_ID,
        "Modal-Secret": TOKEN_SECRET
    }
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.7
    }
    
    try:
        if verbose:
            print(f"\nSending request to Modal API for model: {model_name}")
            
        start_time = datetime.now()
        try:
            response = requests.post(MODAL_URL, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.Timeout:
            print(f"ERROR: Request for {model_name} timed out after {timeout} seconds")
            return None
        except requests.exceptions.ConnectionError:
            print(f"ERROR: Connection error when calling API for {model_name}")
            return None
        end_time = datetime.now()
        total_time_sec = (end_time - start_time).total_seconds()
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from Modal API")
            print(f"Response content: {response.text}")
            return None
            
        # Parse the JSON response
        try:
            result = response.json()
            if verbose:
                print(f"Response received successfully. Length: {len(result.get('response', ''))}")
                print(f"Raw response data: {json.dumps(result, indent=2)}")
                print(f"Response keys from {model_name}: {list(result.keys())}")
                
            # Convert Modal API response format to our OllamaResponse format
            message_content = result.get('response', '')
            
            # Calculate token counts using our estimator
            # Modal API doesn't seem to provide token counts directly
            prompt_tokens = estimate_token_count(prompt)
            response_tokens = estimate_token_count(message_content)
            
            # Calculate timings (rough estimates based on total time)
            prompt_time = total_time_sec * 0.3  # Assume 30% time on prompt
            eval_time = total_time_sec * 0.7    # Assume 70% time on generation
            
            # Convert to nanoseconds
            ns_factor = 1000000000
            
            return OllamaResponse(
                model=model_name,
                message=Message(role="assistant", content=message_content),
                eval_count=response_tokens,
                eval_duration=int(eval_time * ns_factor),
                prompt_eval_count=prompt_tokens,
                prompt_eval_duration=int(prompt_time * ns_factor),
                total_duration=int(total_time_sec * ns_factor)
            )
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {response.text[:200]}...")  # Print first 200 chars of response
            return None
            
    except Exception as e:
        print(f"Error during API request: {str(e)}")
        return None

def nanosec_to_sec(nanosec):
    return nanosec / 1_000_000_000

def inference_stats(model_response: OllamaResponse):
    # Handle potential division by zero
    prompt_ts = model_response.prompt_eval_count / max(nanosec_to_sec(model_response.prompt_eval_duration), 0.0001)
    response_ts = model_response.eval_count / max(nanosec_to_sec(model_response.eval_duration), 0.0001)
    total_tokens = model_response.prompt_eval_count + model_response.eval_count
    total_duration = model_response.prompt_eval_duration + model_response.eval_duration
    total_ts = total_tokens / max(nanosec_to_sec(total_duration), 0.0001)
    
    print(f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s
        
        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tTotal tokens: {total_tokens}
        \tPrompt eval time: {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {nanosec_to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
    """)

def test_model_availability(models: List[str], verbose: bool, timeout: int = 30) -> List[str]:
    """Test which models are available by sending a simple prompt to each"""
    available_models = []
    test_prompt = "Reply with just 'OK' if you can see this."
    
    print("Testing model availability...")
    for model in models:
        if verbose:
            print(f"  Testing {model}...")
        response = run_benchmark(model, test_prompt, verbose=False, timeout=timeout)
        if response:
            available_models.append(model)
            print(f"  ✓ {model} is available")
        else:
            print(f"  ✗ {model} is not available")
    
    return available_models

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on Modal-hosted Ollama models.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity", default=False)
    parser.add_argument("-m", "--models", nargs="*", default=[], help="List of model names to benchmark (default: test all known models)")
    parser.add_argument("-p", "--prompts", nargs="*", default=[
        "Why is the sky blue?", 
        "Write a report on the financials of Apple Inc.",
        "Explain the steps to solve this math problem: If x² + 6x + 8 = 0, what are the values of x?",
        "Write a Python function to find the nth Fibonacci number using recursion",
        "Create a JSON object representing a product catalog with 3 items",
        "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    ], help="List of prompts to use for benchmarking.")
    parser.add_argument("--token-id", help="Modal authentication token ID (can also set TOKEN_ID env var)")
    parser.add_argument("--token-secret", help="Modal authentication token secret (can also set TOKEN_SECRET env var)")
    parser.add_argument("--test-all", action="store_true", help="Test all potential models before benchmarking", default=False)
    parser.add_argument("--add-models", nargs="*", default=[], help="Additional models to test beyond the default list")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)")
    args = parser.parse_args()
    
    global TOKEN_ID, TOKEN_SECRET
    
    # Override environment variables with command line arguments if provided
    if args.token_id:
        TOKEN_ID = args.token_id
    if args.token_secret:
        TOKEN_SECRET = args.token_secret
        
    # Check for authentication credentials
    if not TOKEN_ID or not TOKEN_SECRET:
        print("ERROR: Modal authentication credentials are required.")
        print("Please provide them using environment variables TOKEN_ID and TOKEN_SECRET")
        print("Or using command line arguments --token-id and --token-secret")
        return
    
    verbose = args.verbose
    prompts = args.prompts
    timeout = args.timeout
    
    # Determine which models to test
    models_to_test = args.models
    
    # If no models specified but test-all is set, test all potential models
    if not models_to_test and (args.test_all or args.add_models):
        test_models = DEFAULT_MODELS + args.add_models
        available_models = test_model_availability(test_models, verbose, timeout=timeout)
        if not available_models:
            print("No models are available. Exiting.")
            return
        models_to_test = available_models
    # If no models specified and not testing all, use default
    elif not models_to_test:
        models_to_test = DEFAULT_MODELS  # Use all default models instead of just one
    
    print(f"\nVerbose: {verbose}\nModels: {models_to_test}\nPrompts: {prompts}\nTimeout: {timeout}s")
    
    benchmarks = {}
    
    for model_name in models_to_test:
        responses = []
        for prompt in prompts:
            if verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
            response = run_benchmark(model_name, prompt, verbose=verbose, timeout=timeout)
            if response:
                responses.append(response)
                if verbose:
                    inference_stats(response)
            else:
                print(f"Failed to get response from {model_name}")
        benchmarks[model_name] = responses
    
    print("\n\n===== BENCHMARK RESULTS =====\n")
    
    # Store aggregate stats for final summary
    summary_stats = {}
    
    for model_name, responses in benchmarks.items():
        if responses:
            print(f"\n{'-'*60}")
            print(f"MODEL: {model_name}")
            print(f"{'-'*60}")
            
            model_total_tokens = 0
            model_total_time = 0
            model_token_rate = 0
            prompt_count = 0
            
            for i, response in enumerate(responses):
                prompt = prompts[i]
                print(f"\nPrompt {i+1}: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
                inference_stats(response)
                
                # Accumulate stats for summary
                total_tokens = response.prompt_eval_count + response.eval_count
                total_time_sec = nanosec_to_sec(response.total_duration)
                model_total_tokens += total_tokens
                model_total_time += total_time_sec
                prompt_count += 1
            
            # Calculate average token rate for this model
            if model_total_time > 0:
                model_token_rate = model_total_tokens / model_total_time
            
            # Store in summary
            summary_stats[model_name] = {
                "total_tokens": model_total_tokens,
                "total_time": model_total_time,
                "token_rate": model_token_rate,
                "prompt_count": prompt_count
            }
        else:
            print(f"\nNo successful responses for {model_name}")
    
    # Print final summary comparing all models
    if summary_stats:
        print(f"\n\n{'-'*90}")
        print(f"BENCHMARK SUMMARY (sorted by token generation speed)")
        print(f"{'-'*90}")
        print(f"{'Model':<20} {'Token Rate':<15} {'Avg. Response':<15} {'Response Size':<15} {'Total Time':<15} {'Prompts':<10}")
        print(f"{'-'*90}")
        
        # Sort by token rate (fastest first)
        for model, stats in sorted(summary_stats.items(), key=lambda x: x[1]["token_rate"], reverse=True):
            avg_response_size = stats['total_tokens'] / max(stats['prompt_count'], 1)
            print(
                f"{model:<20} "
                f"{stats['token_rate']:.2f} t/s      "
                f"{avg_response_size:.1f} tokens    "
                f"{stats['total_tokens']:<15} "
                f"{stats['total_time']:.2f}s          "
                f"{stats['prompt_count']:<10}"
            )
        
        print(f"{'-'*90}")
        print("Note: Timings include network latency and are measured from client side")
        print("Token counts are estimated based on text length and may differ from actual model tokenization")

if __name__ == "__main__":
    main()
