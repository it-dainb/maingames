FROM vllm/vllm-openai:latest

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8000

CMD vllm serve itdainb/bloomz-1b1-w4g128-auto-gptq --dtype bf16 --api-key token-abc123