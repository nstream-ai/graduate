# Examples

`traces.jsonl` is a tiny, made-up sample in the OpenAI chat format (three conversations, one line
each): the reference model's replies and tool calls with their results. It shows the format and
is far too small to certify anything: a single slice needs about 70 conversations with zero
disagreements at the defaults, and more when many slices are tested at once or the candidate
sometimes disagrees.

## Without any model (a few seconds)

`mock_candidate.py` is a stand-in candidate (standard library only), so you can see a report card
before setting up a model:

```bash
python examples/mock_candidate.py &
graduate certify examples/traces.jsonl --base-url http://localhost:8000/v1 --model mock --out report/
cat report/card.md
kill %1        # stop the stand-in
```

## With a real model

To run a small model with Ollama or llama.cpp, including memory, speed and thinking models on a
laptop, see [Run on a laptop](https://github.com/nstream-ai/graduate/blob/main/docs/laptop.md).

Use a new `--out` folder for each candidate (or pass `--fresh`): `proposals.jsonl` there caches
the candidate's answers, so a rerun resumes and retries only the requests that failed.

Add `--reference-base-url ... --reference-model ...` to re-ask your reference model on a sample of
states and see where it disagrees with itself.
