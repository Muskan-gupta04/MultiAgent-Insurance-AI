# Insurance Multi-Agent System (LangGraph + Gemini)

A modular Python implementation of an insurance support multi-agent system using:
- LangGraph for orchestration
- Google Gemini (via OpenAI-compatible endpoint) for LLM/tool-calling
- ChromaDB for FAQ retrieval (RAG)
- SQLite for structured policy/billing/claims data
- Phoenix/OpenTelemetry for tracing

## Current Architecture

Agents:
- `supervisor_agent`
- `policy_agent`
- `billing_agent`
- `claims_agent`
- `general_help_agent`
- `human_escalation_agent`
- `final_answer_agent`

Flow:
1. Supervisor routes the request
2. Specialist agent handles domain work
3. Supervisor decides to continue, escalate, or end
4. Final answer agent returns clean user response

## Project Structure

```text
insurance_multi-agent-system/
+-- src/
¦   +-- setup.py            # env loading + tracing decorator
¦   +-- data_pipeline.py    # FAQ ingest + synthetic data + SQLite setup
¦   +-- prompts.py          # all prompt templates
¦   +-- agents.py           # LLM client + tools + agent nodes
¦   +-- workflow.py         # LangGraph graph + run_test_query
+-- run_project.py          # main entrypoint
+-- smoke_test_split.py     # compile-only smoke test
+-- requirements.txt
+-- enhanced_workflow.mmd
+-- README.md
```

## Requirements

- Python 3.10+
- Internet access for:
  - Hugging Face dataset download (`deccan-ai/insuranceQA-v2`)
  - Gemini API calls

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Environment Variables

Create `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:4317
```

Notes:
- `GEMINI_MODEL` is optional. Default is `gemini-2.0-flash`.
- `PHOENIX_COLLECTOR_ENDPOINT` is optional, but warnings appear if Phoenix is not running.

## Run

Windows PowerShell:

```powershell
$env:PYTHONIOENCODING='utf-8'
python run_project.py
```

This does:
1. Build/load FAQ vector DB in `./chroma_db`
2. Generate synthetic insurance data
3. Recreate/populate `insurance_support.db`
4. Execute a sample query through the graph

## Test / Validation

Compile smoke test:

```powershell
python smoke_test_split.py
```

## Typical Runtime Issues

1. `429 RESOURCE_EXHAUSTED` from Gemini
- Cause: API quota/billing exhausted for your key/project/model.
- Fix: enable billing/quota or use a model/key with available quota.

2. Unicode/emoji console errors on Windows
- Use:
```powershell
$env:PYTHONIOENCODING='utf-8'
```

3. Phoenix exporter warnings (`localhost:4317` unavailable)
- Start Phoenix locally or remove/set `PHOENIX_COLLECTOR_ENDPOINT` appropriately.

## Custom Query

You can run your own query from Python:

```python
from src.data_pipeline import initialize_data_infrastructure
from src.workflow import run_test_query

initialize_data_infrastructure()
run_test_query("What is the premium of my auto insurance policy?")
```

## License

MIT (or project default license if specified elsewhere).
