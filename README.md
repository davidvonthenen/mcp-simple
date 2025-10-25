# Local MCP-Enabled LLM Agent

This project provides a lightweight OpenAI-compatible chat completions API backed by a
local `llama-cpp-python` model. The agent integrates with Model Context Protocol (MCP)
tools and ships with a mock news feed that exposes a `fetch_mock_news` tool. Requests
can therefore drive the local model or allow it to call into the mock news service for
fresh headlines.

## Prerequisites
- Python 3.10+
- A quantised GGUF model accessible on disk (see `LLAMA_MODEL_PATH`)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # update paths and overrides as needed
```

## Usage

### Launch the mock MCP news server (optional but recommended)
```bash
make mcp-server
```
The server exposes the `fetch_mock_news` tool over SSE on `http://127.0.0.1:8765/mcp`.

### Run the agent API
```bash
make agent
```
This starts the OpenAI-style endpoint on `0.0.0.0:8000` (configurable via environment
variables). At startup the agent discovers MCP tools and enables function calling for
them automatically.

### Query the agent
```bash
make query QUESTION="Give me the latest Windsurf headlines"
```
The client sends a simple chat completion request, prints the assistant reply, and
shows any tool calls issued by the model. Adjust generation parameters by editing the
payload in `src/client/client.py` or sending custom requests with `curl`.

## Configuration
All configuration is driven by environment variables (see `.env.example`). Key values:
- `LLAMA_MODEL_PATH`, `LLAMA_CTX`, `LLAMA_N_THREADS`, `LLAMA_N_GPU_LAYERS`,
  `LLAMA_N_BATCH`, `LLAMA_N_UBATCH`, `LLAMA_LOW_VRAM`
- `SERVER_HOST`, `SERVER_PORT`
- `MCP_ENABLED`, `MCP_TARGETS`, `MCP_CONNECT_TIMEOUT_SEC`,
  `MCP_INVOCATION_TIMEOUT_SEC`, `MCP_MAX_TOOL_CALL_DEPTH`

The agent automatically discovers tools declared by each MCP target and exposes them to
the model via OpenAI function-calling metadata.

## Notes
- There is no retrieval-augmented generation pipeline—responses rely on the LLM and
  any MCP tools it calls during the conversation.
- Streaming responses are supported when `stream: true` is supplied in the payload.
- The mock news server logs requests and provides deterministic sample articles for
  experimentation with MCP.
