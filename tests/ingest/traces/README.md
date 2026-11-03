# Trace-store fixtures

| File | Source | Licence |
|---|---|---|
| `adk-genai-semconv-schema-v2.json` | `google/adk-python`, `tests/unittests/telemetry/functional_goldens/agent/experimental-span-only-schema-v2.json` | Apache-2.0 |
| `adk-legacy-call_llm-schema-v1.json` | same repository, `agent/stable-capture-schema-v1.json` (content capture off: requests are `{}`) | Apache-2.0 |
| `adk-grounding.json` | same repository, `agent/grounding.json` | Apache-2.0 |
| `mlflow-v3-trace-chat-model-tools.json` | constructed from MLflow's test material: the trace layout of `tests/tracing/helper.py` (schema v3 field names) and the chat request and response of `ModelTraceExplorer.test-utils.ts` | Apache-2.0 (MLflow) |

`PRESENT` in the ADK goldens is the goldens' placeholder for ids.
