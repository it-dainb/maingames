# Exercise 1 - Chatbot for Youtube video

This video-based chatbot is developed using Streamlit, LlamaIndex, and VideoDB. It empowers users to interact with video content by converting both audio and visual information into searchable indexes, then retrieving and synthesizing the most relevant segments to deliver concise responses.

## Running the Application

### Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

### Starting the App

Launch the Streamlit application by running:

```bash
streamlit run app.py
```

## Overview

- **Objective:**  
  Provide an interactive interface where users can query video content by targeting both the spoken words and the visual scenes.

- **Core Components:**  
  - **User Interface:** Developed with Streamlit to facilitate video uploads or YouTube URL entries, and to display interactive chat messages.  
  - **VideoDB Integration:** Handles the upload and indexing of video files, creating separate indexes for spoken words and visual scenes.  
  - **Query Processing:** Uses an OpenAI model to divide the user's query into two parts—one targeting the transcript and the other focusing on scene details.  
  - **Retrieval & Synthesis:** Employs LlamaIndex retrievers along with a response synthesizer to merge results from both indexes into a coherent answer.

## Pipeline Architecture

1. **Video Ingestion & Indexing:**  
   - **Input:** The user supplies a video via a YouTube link or by uploading a local file.
   - **Process:** The video is uploaded to VideoDB and two indexes are created:
     - **Spoken Words Index:** Transcribes and indexes the audio portion into searchable text.
     - **Scene Index:** Detects and indexes key visual scenes, along with detailed descriptions.

2. **Query Transformation:**  
   - **Splitting the Query:** The application separates the input query into two distinct components:
     - **Spoken Query:** Designed to search the audio transcript.
     - **Visual Query:** Aimed at retrieving relevant scene descriptions.
   - **Mechanism:** An OpenAI LLM prompt is used to perform this split.

3. **Retrieval:**  
   - **Spoken Retriever:** Performs semantic search on the transcript index.
   - **Scene Retriever:** Searches the scene index to extract pertinent visual data.

4. **Response Synthesis:**  
   - **Aggregation:** The results from both the spoken and scene retrievers are merged.
   - **Generation:** A compact synthesizer (powered by LlamaIndex) crafts a unified answer based on the aggregated data.

5. **User Interaction:**  
   - **Display:** The synthesized answer is shown within the chat interface.
   - **State Management:** The application maintains session state to facilitate an ongoing conversation.

## Key Technologies

- **Streamlit:** Used to build the web-based interface, enabling API key inputs, file uploads, and chat interactions.
- **VideoDB:** Responsible for video uploads, indexing of spoken words and scenes, and retrieval functionalities.
- **LlamaIndex:** Manages the retrieval-augmented generation workflow, incorporating semantic search and response synthesis.
- **OpenAI API:** Powers the query splitting mechanism and supports answer generation with its LLM capabilities.

## Example Workflow

1. **Initialization:**  
   The user enters their OpenAI and VideoDB API keys in the Streamlit sidebar.

2. **Video Selection:**  
   - The user chooses a video source (either by providing a YouTube URL or uploading a file).
   - The selected video is uploaded to VideoDB, clearing any previous session data if needed.

3. **Indexing:**  
   Upon clicking the **Load Data** button:
   - The application indexes the spoken content (transcription of the audio).
   - It also indexes visual scenes through shot-based extraction.

4. **Query Execution:**  
   - The user submits a query via the chat input.
   - The query is divided into spoken and visual components.
   - The system retrieves the corresponding content from each index.
   - Finally, a synthesized answer is generated and displayed in the chat interface.


# Exercise 2 - LLM deployment

This solution applies advanced quantization techniques using AutoRound and GPTQModel to compress and accelerate large language models for low-bit inference. In our exercise, we compared a baseline FP16 model (bigscience/bloomz-1b1) with a quantized 4-bit version (itdainb/bloomz-1b1-w4g128-auto-gptq).

## Techniques and Evaluation

**Quantization Technique:**  
- **AutoRound Algorithm:**  
  Uses sign gradient descent to fine-tune the rounding and minmax values of model weights over 200 steps. This method competes with recent quantization approaches without incurring additional inference overhead.  
- **GPTQModel Toolkit:**  
  Implements various quantization backends (e.g., TRITON, MARLIN, BITBLAS) to optimize inference speed on CPU/GPU. The quantized model not only supports GPTQ but is also designed to easily integrate additional methods like QQQ.

**Evaluation Metrics:**  
Our evaluations (using lm_eval and internal benchmarks) produced the following comparative metrics:

### Model Performance Comparison

| Models         | Task             | Metric | Value   | Notes                   |
|----------------|------------------|--------|---------|-------------------------|
| **FP16**       | xnli_en          | acc    | 0.4811  | Baseline accuracy       |
|                | xstorycloze_en   | acc    | 0.6446  | Runtime: 13:11          |
|                | xwinograd_en     | acc    | 0.7286  |                         |
| **GPTQ 4-bit** | xnli_en          | acc    | 0.4952  | Slight improvement      |
|                | xstorycloze_en   | acc    | 0.6406  | Runtime: 15:02          |
|                | xwinograd_en     | acc    | 0.7256  |                         |

### Performance Metrics Comparison

| Metric                   | FP16     | GPTQ 4-bit |
|--------------------------|----------|------------|
| **p50_total_tps**        | 52.813   | 79.552     |
| **p90_total_tps**        | 120.742  | 119.646    |
| **p50_decode_tps**       | 22.992   | 31.095     |
| **p90_decode_tps**       | 33.487   | 2.375      |
| **p50_ttft_seconds**     | 0.002    | 0.003      |
| **p90_ttft_seconds**     | 0.003    | 0.011      |
| **max_gpu_memory_mb**    | 2232.0   | 1258.0     |
| **p90_gpu_memory_mb**    | 2232.0   | 1258.0     |
| **max_gpu_utilization**  | 51.0%    | 49.0%      |
| **p90_gpu_utilization**  | 48.0%    | 42.7%      |

*Notes:*
- **VRAM:** The FP16 model consumes about 2232 MB at peak, whereas the quantized GPTQ 4-bit model uses roughly 1258 MB.
- **Speed:** The tokens-per-second metrics indicate that the quantized model (p50_total_tps of ~79.6) is faster in throughput compared to the FP16 baseline (~52.8 TPS) under typical conditions.
- **Latency:** The time to first token (TTFT) is very low in both cases (<0.011 seconds), confirming real-time responsiveness.

## API RESTful Docker Deployment

We provide a Dockerfile for deploying the quantized model as a RESTful API. The base image used is `vllm/vllm-openai:latest`, which supports hardware acceleration (CUDA, ROCm, or IPEX on CPU).

vLLM provides an HTTP server that implements OpenAI's [Completions API](https://platform.openai.com/docs/api-reference/completions), [Chat API](https://platform.openai.com/docs/api-reference/chat), and more!

### Running the Docker Container

1. **Build the Docker Image:**

   ```bash
   docker build -t quantized-llm-api .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 8000:8000 quantized-llm-api
   ```
To call the server, you can use the [official OpenAI Python client](https://github.com/openai/openai-python), or any other HTTP client.

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```
#### Tip
vLLM supports some parameters that are not supported by OpenAI, `top_k` for example.
You can pass these parameters to vLLM using the OpenAI client in the `extra_body` parameter of your requests, i.e. `extra_body={"top_k": 50}` for `top_k`.

### Supported APIs

We currently support the following OpenAI APIs:

- [Completions API](#completions-api) (`/v1/completions`)
  - Only applicable to [text generation models](../models/generative_models.md) (`--task generate`).
  - *Note: `suffix` parameter is not supported.*
- [Chat Completions API](#chat-api) (`/v1/chat/completions`)
  - Only applicable to [text generation models](../models/generative_models.md) (`--task generate`) with a [chat template](#chat-template).
  - *Note: `parallel_tool_calls` and `user` parameters are ignored.*
- [Embeddings API](#embeddings-api) (`/v1/embeddings`)
  - Only applicable to [embedding models](../models/pooling_models.md) (`--task embed`).
- [Transcriptions API](#transcriptions-api) (`/v1/audio/transcriptions`)
  - Only applicable to Automatic Speech Recognition (ASR) models (OpenAI Whisper) (`--task generate`).

In addition, we have the following custom APIs:

- [Tokenizer API](#tokenizer-api) (`/tokenize`, `/detokenize`)
  - Applicable to any model with a tokenizer.
- [Pooling API](#pooling-api) (`/pooling`)
  - Applicable to all [pooling models](../models/pooling_models.md).
- [Score API](#score-api) (`/score`)
  - Applicable to embedding models and [cross-encoder models](../models/pooling_models.md) (`--task score`).
- [Re-rank API](#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)
  - Implements [Jina AI's v1 re-rank API](https://jina.ai/reranker/)
  - Also compatible with [Cohere's v1 & v2 re-rank APIs](https://docs.cohere.com/v2/reference/rerank)
  - Jina and Cohere's APIs are very similar; Jina's includes extra information in the rerank endpoint's response.
  - Only applicable to [cross-encoder models](../models/pooling_models.md) (`--task score`).
