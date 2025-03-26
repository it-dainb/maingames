# Use the latest vLLM OpenAI image as the base
FROM vllm/vllm-openai:latest

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port
EXPOSE 8000

# Set the command to run the vLLM server with the correct syntax
CMD ["vllm", "serve", "itdainb/bloomz-1b1-w4g128-auto-gptq", "--dtype", "bf16", "--api-key", "token-abc123"]
