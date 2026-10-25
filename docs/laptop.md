# Run on a laptop

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint and tool calling works as
a candidate. No API key is sent unless you pass `--api-key-env`, and a local server needs none.
Graduate always asks at temperature 0, so reruns give the same answers where the server is
deterministic.

## Ollama

```bash
brew services start ollama           # macOS; elsewhere run `ollama serve` in another terminal
ollama pull qwen3.5:2b               # any model tagged "tools": https://ollama.com/search?c=tools
graduate certify examples/traces.jsonl --base-url http://localhost:11434/v1 \
  --model qwen3.5:2b --concurrency 1 --out report-qwen3.5-2b/
ollama stop qwen3.5:2b               # unload it when the run is done
```

`--model` must match the pulled tag exactly (`qwen3.5:2b`, not `qwen3.5`).

## llama.cpp

```bash
llama-server -m <model>.gguf --jinja --port 8080
graduate certify examples/traces.jsonl --base-url http://localhost:8080/v1 \
  --model <model> --concurrency 1 --out report-llamacpp/
```

Use any GGUF model file whose chat template supports tools.

## Memory and speed

Measured on an 8 GB Apple M2 (the model runs on the Mac's GPU, through Ollama):

| Model size | Time per call | Fits in 8 GB |
| --- | --- | --- |
| 0.6B | about 0.6 s | yes |
| 2B | about 4 s | yes |
| 4B | about 13 s | no: free memory fell to 10% while the model loaded, before it answered |

So on 8 GB, use models of 2B or less; a 4B needs more memory. Settings for a laptop:

- `--concurrency 1`. The default, 8, is for a server.
- One model loaded at a time. Keep Ollama's default keep-alive during a run (unloading after
  every request makes it reload the model on every call, which is slower and spikes memory), and
  run `ollama stop <model>` when the run is done, also between reruns of the same model.
- Close other apps and watch memory (Activity Monitor). If memory runs low, stop the run: a rerun
  of the same command reuses the cached answers, and the stopped run leaves `card.md` saying so.
- Size runs by calls: a 1,000-call file at 4 s per call is a little over an hour.
- Disk: each model is a download of a few GB (`qwen3.5:2b` 3.1 GB, `qwen3.5:4b` 4.0 GB). If you
  pulled a model only for this, `ollama rm <model>` when done.

`--limit 20` tries the first 20 conversations first.

## Thinking models

Qwen3 and Qwen3.5 think before answering by default, and Ollama has no tag of Qwen3.5 without
thinking. In the default `--prompt-mode chat`, Graduate sends the server's defaults, so thinking
stays on: it reads only the final answer and tool calls, but thinking is slower and counts
against the 1,024-token reply budget. `--prompt-mode completions` renders the model's chat
template with thinking off; it sends token ids to `/completions`, which vLLM accepts (other
servers may not).

## Context length

Each call sends the conversation so far. Ollama's default context (4,096 tokens) is enough for
the examples; longer traces need a larger one (`OLLAMA_CONTEXT_LENGTH`), or Ollama drops the start
of the conversation without saying so.

## Comparing two candidates

Run `certify` once per candidate into separate folders, on the same traces file with the same
options (`--limit` included, so both cards cover the same conversations), and compare the cards
slice by slice. There is no command that diffs two cards. `graduate compare` is for task-success
files (one `{"task_id", "success"}` row per task), not for report cards.

The example file has 3 conversations, so two candidates will rarely differ on it. A comparison
needs your own traces, and a slice needs about 70 conversations to certify.

## The stand-in candidate

`examples/mock_candidate.py` (standard library only) answers like a candidate so you can see a
card with no model. Started with `&`, it keeps running in the background: `kill %1` in the same
terminal stops it.
