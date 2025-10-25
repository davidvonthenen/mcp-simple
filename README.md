# Local Vector RAG Agent (OpenSearch + llama.cpp)

This project provides an end-to-end Retrieval Augmented Generation (RAG) workflow built for macOS on Apple Silicon. It combines OpenSearch 3.2 for vector storage, sentence-transformers embeddings, and a local GGUF LLaMA-class model served via `llama-cpp-python[metal]`. A Flask API exposes an OpenAI-compatible `/v1/chat/completions` endpoint, with supporting scripts for ingesting the BBC dataset and querying the service.

## Prerequisites
- Python 3.10+
- OpenSearch 3.2 running locally with security disabled
- BBC dataset checked out to `./bbc` (from [derekgreene/bbc-datasets](https://github.com/derekgreene/bbc-datasets))
- Quantized GGUF model available at the path defined by `LLAMA_MODEL_PATH`

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit LLAMA_MODEL_PATH and other overrides as needed
```

The ingestor and server will create the OpenSearch index with the proper cosine-similarity HNSW mapping if it does not already exist.

## Usage

### Ingest the BBC dataset
```bash
make ingest
```
By default this indexes the contents of `./bbc` into the `bbc` index using batches of 128 documents. Embeddings are normalized for cosine similarity and stored in a `knn_vector` field.

### Run the Flask server
```bash
make agent
```
This launches the OpenAI-style API on `0.0.0.0:8000` (override via env vars or `.env`). The default target enables the mock MCP
tooling and automatically connects to the bundled news server over stdio.

### Query the server
```bash
make query Q="What is the latest on 3G spectrum auctions?"
```
The client prints the assistant response, retrieved document metadata, and the full JSON response. You can adjust RAG parameters (`k`, `num_candidates`) or generation controls (`temperature`, `top_p`, `max_tokens`) by editing the client payload or sending custom requests.

## Configuration
All configuration is driven by environment variables (see `.env.example`). Key settings include:
- `OPENSEARCH_HOST`, `OPENSEARCH_PORT`, `OPENSEARCH_INDEX`
- `EMBEDDING_MODEL_NAME` (default `thenlper/gte-small`, 384-D embeddings)
- `EMBEDDING_DIMENSION` (override only if your embedding model uses a non-default dimension)
- `LLAMA_MODEL_PATH`, `LLAMA_CTX`, `LLAMA_N_THREADS`, `LLAMA_N_GPU_LAYERS`
- `RAG_TOP_K`, `RAG_NUM_CANDIDATES`
- `SERVER_HOST`, `SERVER_PORT`

To switch to a different embedding model, update `EMBEDDING_MODEL_NAME` and adjust `EMBEDDING_DIMENSION` if necessary. Embeddings are always normalized to ensure cosine similarity matches OpenSearch's `cosinesimil` space.

## MCP tooling and manual validation

The agent can coordinate Model Context Protocol tools that expose OpenAI-style function schemas. The repository includes a mock
news provider to validate end-to-end tool execution.

1. **Start OpenSearch** (see Docker instructions in the project root).
2. **Ingest the BBC dataset**
   ```bash
   make ingest
   ```
3. **Launch the mock MCP server (SSE transport)**
   ```bash
   make mcp-server
   ```
   Leave this running in a separate terminal. It exposes the `fetch_mock_news` tool over SSE by default and logs incoming entities/prompts.
4. **Run the agent with MCP enabled**
   ```bash
   MCP_ENABLED=1 MCP_TARGETS="sse:http://127.0.0.1:8765/mcp" make agent
   ```
   On startup the agent logs discovered tools and registers them with the local LLM.
5. **Query the agent**
   ```bash
   python -m src.3_rag_client.client --question "Give me two recent articles about climate and cite your sources."
   ```
   The assistant should emit a `tool_call` for `fetch_mock_news`, the agent forwards the request to the MCP server, and the
   returned articles are incorporated into the final response.

Additional transports can be specified via `MCP_TARGETS`, e.g. `stdio:python -m src.mock_mcp_server.news_server --stdio`. The agent obeys the `MCP_*`
environment variables documented in `src/common/config.py`:

- `MCP_ENABLED` – toggle MCP integration (default `0`)
- `MCP_TARGETS` – comma-separated list of stdio or SSE endpoints
- `MCP_CONNECT_TIMEOUT_SEC` / `MCP_INVOCATION_TIMEOUT_SEC`
- `MCP_MAX_TOOL_CALL_DEPTH` – limit recursive tool usage

## Notes
- Retrieval uses pure vector k-NN search against OpenSearch with HNSW (`knn_vector`) and cosine similarity—no hybrid queries are performed.
- Context snippets are truncated to stay within the LLM context window.
- The server returns OpenAI-compatible responses with an additional `rag_context` block listing retrieved documents.
- MCP tools are discovered at startup and exposed to the model through OpenAI function-calling metadata. Tool invocations are
  logged with latency metrics for observability.
