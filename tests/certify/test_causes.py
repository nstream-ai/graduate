from graduate.certify.causes import CAUSES, NORMALISER_VERSION, cause, first_difference, redact

REF = [{"name": "get_order", "arguments": {"order_id": "#W1000"}}]


def call(name="get_order", **args):
    return [{"name": name, "arguments": args}]


def test_missing_hash_is_format_only_not_a_wrong_value():
    assert cause(call(order_id="W1000"), REF) == "format-only argument"
    assert cause(call(order_id="#W2000"), REF) == "wrong argument value"


def test_causes_are_assigned_first_match_in_order():
    assert cause([], REF) == CAUSES[0]  # replied instead of acting
    assert cause(REF, []) == CAUSES[1]  # acted instead of replying
    assert cause(call("cancel_order", order_id="#W1000"), REF) == "wrong tool"
    assert cause(REF + REF, REF) == "wrong tool"  # a different number of calls is a wrong tool
    assert cause(call(id="#W1000"), REF) == "missing or extra argument"


def test_numbers_and_separators_normalise():
    ref = [{"name": "pay", "arguments": {"amount": 12.5, "phone": "555-123-2002"}}]
    assert cause([{"name": "pay", "arguments": {"amount": "12.50", "phone": "5551232002"}}],
                 ref) == "format-only argument"  # fmt: skip


def test_arguments_as_a_json_string_are_parsed():
    assert cause([{"name": "get_order", "arguments": '{"order_id": "W1000"}'}], REF) == (
        "format-only argument"
    )


def test_example_names_the_first_differing_argument_and_redaction_keeps_the_format():
    e = first_difference(call(order_id="W1000"), REF)
    assert (e["tool"], e["argument"]) == ("get_order", "order_id")
    assert (redact(e["reference_value"]), redact(e["candidate_value"])) == ("#W0000", "W0000")
    assert redact("jo.doe@mail.com") == "x@x.x"
    assert NORMALISER_VERSION == "1"  # changing the normaliser changes every card: bump it
