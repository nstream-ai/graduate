"""Training data, encoding and decoding. Encoding tests use cached tokenizers only (no network)
and are skipped when the cache or the `train` group is missing."""

from pathlib import Path

import pytest

from graduate.ingest import convert_messages
from graduate.train import build_examples, parse_completion

TOOLS = [{"type": "function", "function": {"name": "get_order", "parameters": {"type": "object"}}}]


def session():
    return convert_messages(
        [
            {"role": "system", "content": "Policy: verify first."},
            {"role": "user", "content": "where is order 7?"},
            {"role": "assistant", "content": "", "reasoning_content": "look it up"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_order", "arguments": '{"id": 7}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "shipped"},
            {"role": "assistant", "content": "It shipped."},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "function_call": {"name": "get_order", "arguments": "{broken"}},
        ],
        source="t", tools=TOOLS,
    )  # fmt: skip


def test_one_example_per_teacher_call_with_merged_target() -> None:
    ex, skipped = build_examples([session()])
    assert len(ex) == 2 and skipped["failed_target"] == 1  # the broken call is not taught
    first, second = ex
    assert first.messages[0] == {"role": "system", "content": "Policy: verify first."}
    assert first.messages[-1]["role"] == "user"
    assert first.target["tool_calls"][0]["function"] == {
        "name": "get_order",
        "arguments": {"id": 7},
    }
    assert "reasoning_content" not in first.target
    assert second.target == {"role": "assistant", "content": "It shipped."}
    assert [m["role"] for m in second.messages] == ["system", "user", "assistant", "tool"]
    assert first.tools[0]["function"]["name"] == "get_order"


def test_reasoning_is_opt_in() -> None:
    ex, _ = build_examples([session()], reasoning=True)
    assert ex[0].target["reasoning_content"] == "look it up"


@pytest.mark.parametrize(
    ("text", "names", "reply"),
    [
        ('<tool_call>\n{"name": "get_order", "arguments": {"id": 7}}\n</tool_call><|im_end|>',
         ["get_order"], "<|im_end|>"),
        ('<think>hmm</think><tool_call>{"name": "a", "arguments": {}}</tool_call>'
         '<tool_call>{"name": "b", "arguments": {}}</tool_call>', ["a", "b"], ""),
        ('{"name": "get_order", "arguments": {"id": 7}}', ["get_order"], ""),
        ("It shipped yesterday.", [], "It shipped yesterday."),
        ('<tool_call>{"name": "get_order", "arguments": {"id": ', ["get_order"], ""),
    ],
)  # fmt: skip
def test_parse_completion(text: str, names: list[str], reply: str) -> None:
    got_reply, calls = parse_completion(text)
    assert [c.name for c in calls] == names and got_reply == reply
    if text.endswith('{"id": '):
        assert calls[0].parse == "failed"  # truncated generation counts as an invalid call


def _tokenizer(name: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"{name} tokenizer not cached")


def test_qwen_template_masks_everything_but_the_target() -> None:
    from graduate.train import encode, ensure_template

    tok = _tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    ensure_template(tok)
    ex, _ = build_examples([session()])
    enc = encode(ex[0], tok, max_len=4096)
    target_ids = [t for t in enc.labels if t != -100]
    assert tok.decode(target_ids).startswith("<tool_call>")
    assert '"get_order"' in enc.target_text and enc.target_text.endswith("<|im_end|>\n")
    assert len(enc.input_ids) == len(enc.labels)
    short = encode(ex[1], tok, max_len=8)  # left-truncated prompt, intact target
    assert tok.decode([t for t in short.labels if t != -100]).startswith("It shipped.")


def test_template_that_drops_tool_calls_is_rejected() -> None:
    from graduate.train import TemplateMismatchError, ensure_template, render

    tok = _tokenizer("hf-internal-testing/tiny-random-LlamaForCausalLM")
    ensure_template(tok)
    ex, _ = build_examples([session()])
    with pytest.raises(TemplateMismatchError, match="drops tool call"):
        render(ex[0], tok)
    ensure_template(tok, force_fallback=True)
    assert "<tool_call>" in render(ex[0], tok)[1]


GEMMA_TEMPLATE = Path(__file__).parent / "fixtures" / "gemma4_chat_template.jinja"


def test_gemma_parser_reads_its_own_syntax() -> None:
    from graduate.train.gemma import parse_arguments

    body = '{city:<|"|>Paris, "FR"<|"|>,days:3,opts:{metric:true,tags:[<|"|>a<|"|>,2]},x:null}'
    assert parse_arguments(body) == {
        "city": 'Paris, "FR"', "days": 3, "opts": {"metric": True, "tags": ["a", 2]}, "x": None,
    }  # fmt: skip
    text, calls = parse_completion(
        '<|channel>thought\nlook it up<channel|><|tool_call>call:get_order{id:<|"|>A7<|"|>}'
        '<tool_call|><|tool_call>call:ping{}<tool_call|><turn|>'
    )  # fmt: skip
    assert [(c.name, c.arguments) for c in calls] == [("get_order", {"id": "A7"}), ("ping", {})]
    assert text == ""
    _, cut = parse_completion('<|tool_call>call:get_order{id:<|"|>A')
    assert cut[0].name == "get_order" and cut[0].parse == "failed"


def test_gemma_round_trip_through_the_real_template() -> None:
    """Render a target with Gemma 4's own chat template, then parse it back."""
    jinja2 = pytest.importorskip("jinja2")
    if not GEMMA_TEMPLATE.exists():
        pytest.skip("Gemma 4 template fixture missing")
    from graduate.train.encode import render

    class Tok:  # minimal stand-in: the real tokenizer's apply_chat_template renders this way
        chat_template = GEMMA_TEMPLATE.read_text()

        def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                                tokenize=False, **template_vars):  # fmt: skip
            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                     extensions=["jinja2.ext.loopcontrols"])  # fmt: skip
            env.filters["tojson"] = lambda v, **k: __import__("json").dumps(v, **k)

            def raise_exception(msg):
                raise ValueError(msg)

            env.globals["raise_exception"] = raise_exception
            return env.from_string(self.chat_template).render(
                messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                bos_token="<bos>", **template_vars,
            )  # fmt: skip

    ex, _ = build_examples([session()])
    _, target = render(ex[0], Tok())
    _, calls = parse_completion(target)
    assert [(c.name, c.arguments) for c in calls] == [("get_order", {"id": 7})]


def test_long_prompt_drops_oldest_turns_not_the_policy() -> None:
    from graduate.train import encode, ensure_template
    from graduate.train.data import SFTExample
    from graduate.train.encode import fit_messages, render_prompt

    tok = _tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    ensure_template(tok)
    policy = "POLICY: always verify the user. " * 20
    turns = []
    for i in range(30):
        turns += [{"role": "user", "content": f"question {i} " + "x " * 40},
                  {"role": "assistant", "content": f"answer {i}"}]  # fmt: skip
    messages = [{"role": "system", "content": policy}, *turns[:-1]]
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Look up.",
              "parameters": {"type": "object", "properties": {}}}}]  # fmt: skip
    fitted = fit_messages(messages, tools, tok, 1200)
    prompt = render_prompt(fitted, tools, tok)
    assert len(tok(prompt, add_special_tokens=False)["input_ids"]) <= 1200
    assert fitted[0]["content"] == policy and "lookup" in prompt  # policy and tools kept
    assert fitted[1]["role"] == "user" and fitted[-1] == messages[-1]  # newest turn kept
    assert len(fitted) < len(messages)
    ex = SFTExample("s", 0, messages, tools, {"role": "assistant", "content": "answer 29"})
    enc = encode(ex, tok, max_len=1200)
    assert len(enc.input_ids) <= 1200 and "POLICY" in tok.decode(enc.input_ids[:300])


QWEN35_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_user_id_by_name_zip",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "zip": {"type": "string"},
                    "count": {"type": "integer"},
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
]


def test_parse_completion_reads_qwen35_xml_calls_typed_by_schema() -> None:
    text = (
        "Let me look you up.\n<tool_call>\n<function=find_user_id_by_name_zip>\n"
        "<parameter=first_name>\nYusuf\n</parameter>\n<parameter=zip>\n19122\n</parameter>\n"
        "<parameter=count>\n3\n</parameter>\n"
        '<parameter=item_ids>\n["1", "2"]\n</parameter>\n</function>\n</tool_call>'
    )
    reply, calls = parse_completion(text, QWEN35_TOOLS)
    assert reply == "Let me look you up."
    assert len(calls) == 1 and calls[0].parse == "ok"
    assert calls[0].name == "find_user_id_by_name_zip"
    assert calls[0].arguments == {"first_name": "Yusuf", "zip": "19122", "count": 3,
                                  "item_ids": ["1", "2"]}  # fmt: skip


def test_parse_completion_qwen35_without_schema_keeps_numbers_as_text() -> None:
    text = (
        "<tool_call>\n<function=f>\n<parameter=zip>\n19122\n</parameter>\n</function>\n</tool_call>"
    )
    _, calls = parse_completion(text)
    assert calls[0].arguments == {"zip": "19122"}


def test_parse_completion_qwen35_truncated_call_fails() -> None:
    _, calls = parse_completion("<tool_call>\n<function=f>\n<parameter=zip>\n191")
    assert calls[0].name == "f" and calls[0].parse == "failed"
