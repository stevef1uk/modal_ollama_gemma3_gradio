#!/usr/bin/env python3
"""
LLM Model Comparison Tool

This script tests multiple language models through the Modal API and compares their responses.
It measures response time, formats the output in a readable table, and saves both
summarized and full results to files for analysis.

Features:
- Tests multiple models with the same set of prompts
- Measures response time for each model/prompt combination
- Handles errors gracefully with informative messages
- Saves results in both human-readable text and machine-readable JSON formats
- Uses environment variables for authentication
- Configurable timeout to prevent hanging on slow responses

Usage:
1. Set your Modal API credentials as environment variables:
   export TOKEN_ID=your-token-id
   export TOKEN_SECRET=your-token-secret

2. Run the script:
   ./run_multiple_models.py

3. Review the results in:
   - Terminal output (summary table)
   - model_comparison_results.txt (formatted text results)
   - model_comparison_results.json (structured data for further analysis)

Author: [Your Name]
Date: [Current Date]
"""

import requests
import json
import time
import os
from tabulate import tabulate

# API Configuration
API_URL = "https://stevef1uk--ollama-api-api.modal.run/"

# Get authentication tokens from environment variables
TOKEN_ID = os.environ.get("TOKEN_ID")
TOKEN_SECRET = os.environ.get("TOKEN_SECRET")

# Validate that authentication tokens are available
if not TOKEN_ID or not TOKEN_SECRET:
    print("Error: TOKEN_ID and TOKEN_SECRET environment variables must be set.")
    print("Please set them with:")
    print("  export TOKEN_ID=your-token-id")
    print("  export TOKEN_SECRET=your-token-secret")
    exit(1)

# Models to test - can be customized as needed
models = ["gemma3:27b", "llama3:8b", "phi3:14b", "mistral:7b", "DeepSeek-R1:14b", "llama3.3"]

# Test prompts - add or modify as needed for your evaluation
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms",
    "Write a short poem about the ocean"
]

# Data structures to store results
results = []  # For the summary table
full_responses = {}  # For storing complete responses

# Test each model with each prompt
for model in models:
    # Initialize storage for this model's responses
    if model not in full_responses:
        full_responses[model] = {}
        
    for prompt in prompts:
        print(f"Testing {model} with prompt: {prompt[:30]}...")
        
        # Measure response time
        start_time = time.time()
        try:
            # Send request to the API
            response = requests.post(
                API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Modal-Key": TOKEN_ID,
                    "Modal-Secret": TOKEN_SECRET
                },
                json={
                    "prompt": prompt,
                    "temperature": 0.7,
                    "model": model
                },
                timeout=120  # 2-minute timeout to prevent hanging
            )
            end_time = time.time()
            
            # Process successful responses
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                
                # Store the complete, untruncated response
                full_responses[model][prompt] = response_text
                
                # Create a preview for the table display
                preview_length = 100
                response_preview = response_text[:preview_length]
                if len(response_text) > preview_length:
                    response_preview += "..."
                
                # Replace newlines with spaces for better table formatting
                table_preview = response_preview.replace("\n", " ")
                
                # Add to results table
                results.append([
                    model,
                    prompt[:30] + "..." if len(prompt) > 30 else prompt,
                    f"{end_time - start_time:.2f}s",
                    table_preview
                ])
            else:
                # Handle API errors
                error_msg = f"Error: {response.status_code}"
                try:
                    error_details = response.text[:50]
                    error_msg += f" - {error_details}"
                except:
                    pass
                
                results.append([
                    model,
                    prompt[:30] + "..." if len(prompt) > 30 else prompt,
                    f"{end_time - start_time:.2f}s",
                    error_msg
                ])
                full_responses[model][prompt] = f"ERROR: {error_msg}"
        except Exception as e:
            # Handle network or other exceptions
            end_time = time.time()
            error_msg = f"Error: {str(e)}"
            results.append([
                model,
                prompt[:30] + "..." if len(prompt) > 30 else prompt,
                f"{end_time - start_time:.2f}s",
                error_msg
            ])
            full_responses[model][prompt] = f"ERROR: {error_msg}"

# Display results table in the terminal
print("\nResults:")
print(tabulate(
    results,
    headers=["Model", "Prompt", "Time", "Response Preview"],
    tablefmt="grid"
))

# Save detailed results to a text file
with open("model_comparison_results.txt", "w") as f:
    f.write("Model Comparison Results\n")
    f.write("======================\n\n")
    f.write(tabulate(
        results,
        headers=["Model", "Prompt", "Time", "Response Preview"],
        tablefmt="grid"
    ))
    f.write("\n\nFull Responses:\n\n")
    
    # Write complete, untruncated responses
    for model in models:
        f.write(f"\n\n{'='*20} MODEL: {model} {'='*20}\n\n")
        for prompt in prompts:
            f.write(f"\nPROMPT: {prompt}\n")
            f.write(f"{'-'*80}\n")
            if prompt in full_responses.get(model, {}):
                f.write(full_responses[model][prompt])
            else:
                f.write("No response collected")
            f.write(f"\n{'-'*80}\n")

# Save structured data as JSON for programmatic analysis
with open("model_comparison_results.json", "w") as f:
    json.dump(full_responses, f, indent=2)

print(f"Results saved to model_comparison_results.txt")
print(f"Full responses saved to model_comparison_results.json")
