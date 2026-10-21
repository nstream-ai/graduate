"""Render examples with the student's own chat template and mask the loss to the target.

The prompt is the template applied to the input with a generation prompt; the target is what the
template appends for the assistant turn. If a template does not render the full conversation as
"prompt + target" (some templates rewrite earlier turns), the example is rejected and counted,
never silently mis-masked. Tokenizers without a template get a ChatML fallback that lists tools
in the system message and writes calls as Hermes <tool_call> blocks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from graduate.train.data import SFTExample

IGNORE = -100

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


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    labels: list[int]
    prompt_text: str
    target_text: str


class TemplateMismatchError(ValueError):
    pass


def ensure_template(tokenizer: Any, *, force_fallback: bool = False) -> None:
    if force_fallback or not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = CHATML_FALLBACK
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


def render_prompt(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]], tokenizer: Any
) -> str:
    """The prompt alone, as in training (for live serving, where there is no target yet)."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _n_tokens(text: str, tokenizer: Any) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_messages(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]], tokenizer: Any, budget: int
) -> list[dict[str, Any]]:
    """Messages whose rendered prompt fits in `budget` tokens. Over-long conversations lose their
    oldest turns, never the system message (the policy) or the tools, and the kept history starts
    at a user message (no orphaned tool results). A plain left cut of tokens would remove the
    policy and tools first. Used identically in training, evaluation and live serving."""
    if _n_tokens(render_prompt(messages, tools, tokenizer), tokenizer) <= budget:
        return messages
    head = messages[:1] if messages and messages[0]["role"] == "system" else []
    rest = messages[len(head) :]
    starts = [i for i, m in enumerate(rest) if m["role"] == "user" and i > 0]
    lo, hi, best = 0, len(starts) - 1, None  # the earliest start that fits keeps the most history
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = head + rest[starts[mid] :]
        if _n_tokens(render_prompt(cand, tools, tokenizer), tokenizer) <= budget:
            best, hi = cand, mid - 1
        else:
            lo = mid + 1
    if best is not None:
        return best
    return head + rest[starts[-1] :] if starts else messages  # the caller's token cut is the net


def render(example: SFTExample, tokenizer: Any) -> tuple[str, str]:
    # enable_thinking=False: hybrid-reasoning templates (Qwen3) would otherwise put an empty
    # <think></think> block in every target; templates without the switch ignore it.
    kwargs: dict[str, Any] = {"tools": example.tools or None, "tokenize": False,
                              "enable_thinking": False}  # fmt: skip
    prompt = render_prompt(example.messages, example.tools, tokenizer)
    full = tokenizer.apply_chat_template([*example.messages, example.target], **kwargs)
    if not full.startswith(prompt):
        raise TemplateMismatchError("template does not render conversation as prompt + target")
    target = full[len(prompt) :]
    # Some templates silently drop tool calls (or content): the loss would then teach nothing.
    for call in example.target.get("tool_calls", []):
        if call["function"]["name"] not in target:
            raise TemplateMismatchError(f"template drops tool call {call['function']['name']!r}")
    content = (example.target.get("content") or "").strip()
    if content and content[:50] not in target:
        raise TemplateMismatchError("template drops the reply text")
    return prompt, target


def encode(
    example: SFTExample, tokenizer: Any, max_len: int = 4096, stop_ids: set[int] | None = None
) -> Encoded:
    """Token ids with labels only on the target. Long prompts are cut from the left (the oldest
    history goes first); an example whose target alone does not fit is rejected."""
    prompt, target = render(example, tokenizer)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(t_ids) + 1 > max_len:  # drop the oldest turns, keep the policy and tools
        fitted = fit_messages(example.messages, example.tools, tokenizer,
                              max_len - len(t_ids) - 1)  # fmt: skip
        if fitted is not example.messages:
            prompt, target = render(replace(example, messages=fitted), tokenizer)
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # End the target with a stop token unless the template already did (Gemma ends a tool call
    # with <|tool_response>, a model turn with <turn|>; both are in its generation stop list).
    stops = (stop_ids or set()) | ({tokenizer.eos_token_id} - {None})
    if tokenizer.eos_token_id is not None and (not t_ids or t_ids[-1] not in stops):
        t_ids = [*t_ids, tokenizer.eos_token_id]
    if len(t_ids) >= max_len:
        raise ValueError("target longer than max_len")
    p_ids = p_ids[-(max_len - len(t_ids)) :]
    return Encoded(p_ids + t_ids, [IGNORE] * len(p_ids) + t_ids, prompt, target)


def target_summary(example: SFTExample) -> str:
    t = example.target
    calls = [c["function"] for c in t.get("tool_calls", [])]
    return json.dumps({"content": t.get("content", ""), "tool_calls": calls}, default=str)
