# Examples

`traces.jsonl` is a tiny, made-up sample in the OpenAI chat format (three conversations, one line
each): the reference model's replies and tool calls with their results. It is far too small to
certify anything. A single slice needs about 70 conversations with zero disagreements at the
defaults, and more when many slices are tested at once or the candidate sometimes disagrees.

## 1. Without any model (a few seconds)

`mock_candidate.py` is a stand-in candidate (standard library only) so you can see a report card
before setting up a model:

```bash
python examples/mock_candidate.py &
graduate certify examples/traces.jsonl --base-url http://localhost:8000/v1 --model mock --out report/
cat report/card.md
```

## 2. With a small model on your CPU

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint and tool calling works. No
API key is sent unless you pass `--api-key-env`, and a local server needs none.

**Ollama** (macOS, Linux, Windows; https://ollama.com):

```bash
ollama pull <model>          # any small model tagged "tools": https://ollama.com/search?c=tools
graduate certify examples/traces.jsonl --base-url http://localhost:11434/v1 --model <model> --out report-ollama/
```

**llama.cpp** (`llama-server`, with any GGUF model file whose chat template supports tools):

```bash
llama-server -m <model>.gguf --jinja --port 8080
graduate certify examples/traces.jsonl --base-url http://localhost:8080/v1 --model <model> --out report-llamacpp/
```

Small models on a CPU answer slowly. Use `--limit 20` to try a few conversations first, and
`--concurrency 1` if the server handles one request at a time.

Use a new `--out` folder for each candidate (or pass `--fresh`): `report/proposals.jsonl` caches
the candidate's answers, so a rerun resumes and retries only the requests that failed.

Add `--reference-base-url ... --reference-model ...` to re-ask your reference model on a sample of
states and see where it disagrees with itself.
