"""Reading a candidate's reply back as actions: OpenAI tool calls, Qwen XML calls typed by the
tool schema, and Gemma's call syntax."""

import pytest

from graduate.prompting import parse_completion

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


def test_gemma_parser_reads_its_own_syntax() -> None:
    from graduate.prompting.gemma import parse_arguments

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


def test_parse_completion_qwen35_without_schema_keeps_numbers_as_text() -> None:
    text = (
        "<tool_call>\n<function=f>\n<parameter=zip>\n19122\n</parameter>\n</function>\n</tool_call>"
    )
    _, calls = parse_completion(text)
    assert calls[0].arguments == {"zip": "19122"}


def test_parse_completion_qwen35_truncated_call_fails() -> None:
    _, calls = parse_completion("<tool_call>\n<function=f>\n<parameter=zip>\n191")
    assert calls[0].name == "f" and calls[0].parse == "failed"


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


def test_thinking_opened_by_the_prompt_is_dropped_before_parsing():
    # with enable_thinking the template opens <think> in the prompt: the reply has only the close
    text = (
        "the user wants order 7, look it up\n</think>\n\n<tool_call>\n<function=get_order>\n"
        "<parameter=id>\n7\n</parameter>\n</function>\n</tool_call>"
    )
    reply, calls = parse_completion(text, QWEN35_TOOLS)
    assert [c.name for c in calls] == ["get_order"] and "look it up" not in reply
