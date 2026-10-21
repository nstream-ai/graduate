# Example

`traces.jsonl` is a tiny, made-up sample in the OpenAI chat format (three conversations, one line
each): the reference model's replies and tool calls with their results. Your own traces need many
more conversations for any slice to certify: on the order of a hundred conversations per slice
with no disagreement at alpha = 5% (the anytime-valid test pays a little for being valid at every
look).

```bash
# a candidate behind any OpenAI-compatible endpoint, e.g. vLLM
vllm serve Qwen/Qwen3.5-4B --enable-auto-tool-choice --tool-call-parser qwen3_coder &
graduate certify examples/traces.jsonl --preset openai \
    --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-4B --out report/
cat report/card.md
```

Add `--reference-base-url ... --reference-model ...` to re-ask your reference model on a sample of
states and see where it disagrees with itself.
