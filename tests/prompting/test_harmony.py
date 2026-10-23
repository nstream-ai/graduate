"""gpt-oss harmony output read back as actions; the dispatch leaves Qwen and Gemma alone."""

import pytest

from graduate.prompting import parse_completion
from graduate.prompting.harmony import is_harmony, parse_harmony

CALL_CONSTRAIN = (
    '<|channel|>commentary to=functions.get_order <|constrain|>json<|message|>{"order_id": "#W1"}'
    "<|call|>"
)
CALL_PLAIN = '<|channel|>commentary to=functions.get_order<|message|>{"order_id": "#W1"}<|call|>'
CALL_RECIPIENT_FIRST = (
    "<|start|>assistant to=functions.get_order<|channel|>commentary json<|message|>"
    '{"order_id": "#W1"}<|call|>'
)
ANALYSIS = "<|channel|>analysis<|message|>The user wants order #W1. Look it up.<|end|>"


@pytest.mark.parametrize(
    "text",
    [
        CALL_CONSTRAIN,
        CALL_PLAIN,
        CALL_RECIPIENT_FIRST,
        ANALYSIS + "<|start|>assistant" + CALL_CONSTRAIN,
        "assistant" + CALL_CONSTRAIN,
        CALL_CONSTRAIN.removesuffix("<|call|>"),  # the server stripped the stop token
    ],
)
def test_tool_call(text: str) -> None:
    reply, calls = parse_completion(text)
    assert reply == ""
    assert [(c.name, c.arguments, c.parse) for c in calls] == [
        ("get_order", {"order_id": "#W1"}, "ok")
    ]


def test_two_calls_in_one_completion() -> None:
    second = CALL_PLAIN.replace("get_order", "get_user")
    _, calls = parse_completion(
        CALL_CONSTRAIN.replace("<|call|>", "<|end|>") + "<|start|>assistant" + second
    )
    assert [c.name for c in calls] == ["get_order", "get_user"]


@pytest.mark.parametrize(
    "text",
    [
        "<|channel|>final<|message|>Your order has shipped.<|return|>",
        ANALYSIS + "<|start|>assistant<|channel|>final<|message|>Your order has shipped.<|return|>",
        ANALYSIS + "<|start|>assistant<|channel|>final<|message|>Your order has shipped.",
    ],
)
def test_final_reply(text: str) -> None:
    assert parse_completion(text) == ("Your order has shipped.", [])


def test_analysis_only_is_an_empty_reply() -> None:
    assert parse_completion(ANALYSIS) == ("", [])


def test_preamble_is_the_reply_when_there_is_no_final() -> None:
    text = "<|channel|>commentary<|message|>Let me check.<|end|><|start|>assistant" + CALL_PLAIN
    reply, calls = parse_completion(text)
    assert reply == "Let me check." and [c.name for c in calls] == ["get_order"]


@pytest.mark.parametrize(
    "body",
    ['{"order_id": ', "not json at all", ""],  # truncated, garbage, empty
)
def test_malformed_arguments_are_a_failed_call(body: str) -> None:
    _, calls = parse_harmony(f"<|channel|>commentary to=functions.get_order<|message|>{body}")
    assert [(c.name, c.parse) for c in calls] == [("get_order", "failed")]


@pytest.mark.parametrize(
    ("text", "names"),
    [  # other families never reach the harmony parser and parse as before
        ('<tool_call>\n{"name": "get_order", "arguments": {"id": 7}}\n</tool_call>', ["get_order"]),
        (
            "<tool_call>\n<function=get_order>\n<parameter=id>\n7\n</parameter>\n</function>",
            ["get_order"],
        ),
        ("<|tool_call>call:get_order{id:7}<tool_call|>", ["get_order"]),
        ("<|channel>thought\nhmm<channel|>Hello there.", []),
        ("Hello there.", []),
    ],
)
def test_dispatch_keeps_qwen_and_gemma(text: str, names: list[str]) -> None:
    assert not is_harmony(text)
    assert [c.name for c in parse_completion(text)[1]] == names
