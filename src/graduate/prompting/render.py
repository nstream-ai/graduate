"""The prompt a candidate model is shown, rendered the way the research path renders it: the
model's own chat template with the tools, thinking off, fitted to a token budget by dropping the
oldest turns (never the policy or the tools). `graduate certify --prompt-mode completions` uses it
to call the candidate's `/completions` endpoint directly; needs `transformers` (the `completions`
extra)."""

from __future__ import annotations

from typing import Any

CHATML_FALLBACK = (
    "{%- if tools %}<|im_start|>system\n"
    "{%- for m in messages if m.role == 'system' %}{{ m.content }}\n{% endfor -%}"
    "# Tools\n<tools>{% for t in tools %}{{ t | tojson }}{% endfor %}</tools>"
    "<|im_end|>\n{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m.role != 'system' or not tools %}"
    "<|im_start|>{{ m.role }}\n{{ m.content }}"
    "{%- for c in m.tool_calls or [] %}<tool_call>\n{{ c.function | tojson }}\n</tool_call>"
    "{%- endfor %}<|im_end|>\n"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
)


def ensure_template(tokenizer: Any, *, force_fallback: bool = False) -> None:
    if force_fallback or not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = CHATML_FALLBACK
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


def render_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tokenizer: Any,
    thinking: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    """The prompt alone, as in training (for live serving, where there is no target yet).
    `reasoning_effort` ("low" / "medium" / "high") is read by the gpt-oss chat template (its
    default is "medium"); it is passed only when given, so other templates render as before."""
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
        **extra,
    )


def _n_tokens(text: str, tokenizer: Any) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tokenizer: Any,
    budget: int,
    thinking: bool = False,
) -> list[dict[str, Any]]:
    """Messages whose rendered prompt fits in `budget` tokens. Over-long conversations lose their
    oldest turns, never the system message (the policy) or the tools, and the kept history starts
    at a user message (no orphaned tool results). A plain left cut of tokens would remove the
    policy and tools first. Used identically in training, evaluation and live serving."""
    if _n_tokens(render_prompt(messages, tools, tokenizer, thinking), tokenizer) <= budget:
        return messages
    head = messages[:1] if messages and messages[0]["role"] == "system" else []
    rest = messages[len(head) :]
    starts = [i for i, m in enumerate(rest) if m["role"] == "user" and i > 0]
    lo, hi, best = 0, len(starts) - 1, None  # the earliest start that fits keeps the most history
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = head + rest[starts[mid] :]
        if _n_tokens(render_prompt(cand, tools, tokenizer, thinking), tokenizer) <= budget:
            best, hi = cand, mid - 1
        else:
            lo = mid + 1
    if best is not None:
        return best
    return head + rest[starts[-1] :] if starts else messages  # the caller's token cut is the net


def stop_token_ids(model_or_path: Any) -> set[int]:
    """End-of-generation token ids from the model's generation config (one or a list)."""
    try:
        if isinstance(model_or_path, str):
            from transformers import GenerationConfig

            cfg = GenerationConfig.from_pretrained(model_or_path)
        else:
            cfg = model_or_path.generation_config
        eos = cfg.eos_token_id
    except (OSError, AttributeError, ValueError):
        return set()
    return set(eos if isinstance(eos, list) else [eos]) - {None}
