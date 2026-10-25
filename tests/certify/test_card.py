import numpy as np
import pytest

from graduate.certify.card import (
    choose_caps,
    more_sessions_needed,
    plan_text,
    report_card,
    within_cap,
)


def rows(rng: np.random.Generator, n_sessions: int, rate: float, key: str, per: int = 3):
    keys, sess, wrong = [], [], []
    for s in range(n_sessions):
        for _ in range(per):
            keys.append(key)
            sess.append(f"{key}-{s}")
            wrong.append(rng.random() < rate)
    return keys, sess, wrong


def test_clean_slice_graduates_and_bad_slice_does_not() -> None:
    rng = np.random.default_rng(0)
    k1, s1, w1 = rows(rng, 600, 0.0, "easy")
    k2, s2, w2 = rows(rng, 600, 0.20, "hard")
    card = report_card(k1 + k2, s1 + s2, w1 + w2, alpha=0.05, cap=5)
    by = {r.slice: r for r in card.slices}
    assert by["easy"].certified and not by["hard"].certified
    assert by["easy"].upper <= 0.05 and by["hard"].rate > 0.1
    assert card.certified_call_share == pytest.approx(0.5)


def test_boundary_slices_rarely_graduate() -> None:
    """True rate exactly at alpha: certifying would be a false graduation."""
    rng = np.random.default_rng(1)
    false = 0
    reps = 200
    for _ in range(reps):
        k, s, w = rows(rng, 300, 0.05, "b", per=2)
        false += report_card(k, s, w, alpha=0.05, cap=2, plan=False).slices[0].certified
    assert false / reps <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / reps)


def test_cap_limits_the_certified_population() -> None:
    k, s, w = rows(np.random.default_rng(2), 10, 0.0, "x", per=8)
    r = report_card(k, s, w, cap=3).slices[0]
    assert (r.calls, r.certified_population, r.sessions) == (80, 30, 10)


def test_token_share_and_validation() -> None:
    card = report_card(["a", "b"], ["s1", "s2"], [False, False], [3.0, 1.0])
    assert {r.slice: r.token_share for r in card.slices} == {"a": 0.75, "b": 0.25}
    with pytest.raises(ValueError):
        report_card(["a"], ["s1", "s2"], [False])
    with pytest.raises(ValueError):
        report_card(["a"], ["s1"], [False], multiplicity="none")


def test_caps_from_dev_and_coverage_inside_cap() -> None:
    sess = ["a", "a", "a", "b", "c"]
    assert within_cap(sess, 2).tolist() == [True, True, False, True, True]
    assert choose_caps(["x"] * 5, sess, quantile=1.0) == {"x": 3}
    card = report_card(["x"] * 5, sess, [False] * 5, cap={"x": 2})
    r = card.slices[0]
    assert (r.cap, r.certified_population, r.call_share) == (2, 4, 0.8)


def test_small_cap_gains_power_on_short_sessions() -> None:
    rng = np.random.default_rng(3)
    k, s, w = rows(rng, 700, 0.01, "x", per=1)
    loose = report_card(k, s, w, alpha=0.05, cap=20).slices[0]
    tight = report_card(k, s, w, alpha=0.05, cap={"x": 1}).slices[0]
    assert tight.upper < loose.upper and tight.pvalue < loose.pvalue


def test_every_gate_must_pass_and_binding_gate_is_reported() -> None:
    rng = np.random.default_rng(4)
    n = 2000
    keys, sess = ["x"] * n, [f"s{i}" for i in range(n)]
    disagree = rng.random(n) < 0.005
    extra_field = rng.random(n) < 0.04  # good on average, bad in one way
    card = report_card(
        keys, sess, gates={"disagree": disagree, "extra_field": extra_field},
        alpha={"disagree": 0.03, "extra_field": 0.01}, cap=1,
    )  # fmt: skip
    r = card.slices[0]
    assert r.gates["disagree"].pvalue <= 0.05 and not r.certified
    assert r.binding == "extra_field" and r.pvalue == r.gates["extra_field"].pvalue
    alone = report_card(keys, sess, disagree, alpha=0.03, cap=1).slices[0]
    assert alone.certified


def test_gate_arguments_are_validated() -> None:
    with pytest.raises(ValueError):
        report_card(["a"], ["s"])
    with pytest.raises(ValueError):
        report_card(["a"], ["s"], [False], gates={"g": [False]})


def test_holm_option() -> None:
    k, s, w = rows(np.random.default_rng(5), 800, 0.0, "x", per=1)
    card = report_card(k, s, w, multiplicity="holm", cap=1)
    assert card.slices[0].certified and card.multiplicity == "holm"


def test_shielded_calls_leave_the_population_but_count_in_coverage_denominator() -> None:
    k, s, w = rows(np.random.default_rng(6), 700, 0.0, "x", per=1)
    w = list(w)
    w[:70] = [True] * 70  # 10% errors, all caught by the shield
    accepted = [False] * 70 + [True] * 630
    plain = report_card(k, s, w, cap=1).slices[0]
    shielded = report_card(k, s, w, cap=1, accepted=accepted).slices[0]
    assert not plain.certified and shielded.certified
    assert shielded.call_share == 630 / 700


def test_render_says_how_much_more_evidence_and_survives_rate_one() -> None:
    from graduate.certify.card import report_card

    keys = ["a"] * 10 + ["b"] * 10
    sess = [f"s{i}" for i in range(20)]
    wrong = [False] * 10 + [True] * 10  # "a": clean but too small; "b": always wrong
    text = report_card(keys, sess, wrong=wrong, alpha=0.05, delta=0.05).render()
    assert "more conversations" in text


# --- planner: "how many more conversations", with the card's own test ------------


def _two_slices(n: int, bad_every: int | None = None, plan: bool = True):
    """The recorded new-user case: a text slice that always agrees next to a tool slice that
    always disagrees, one call of each per conversation, cap 1."""
    keys, sess, wrong = [], [], []
    for i in range(n):
        keys += ["tool", "text"]
        sess += [f"s{i:04d}"] * 2
        wrong += [True, bool(bad_every and i % bad_every == 0)]
    return report_card(keys, sess, wrong, cap=1, plan=plan)


def _first_pass(lo: int, hi: int) -> int:
    return next(n for n in range(lo, hi) if _text(_two_slices(n, plan=False)).certified)


def _text(card):
    return next(r for r in card.slices if r.slice == "text")


def test_planner_recorded_case_59_62_70():
    # v0.1 printed "~1 more" at 59 and 62; the slice passed only at 70 (uncorrected) and later
    # with the BH correction. The range must now contain the real passing point.
    first = _first_pass(55, 200)
    for n in (59, 62):
        plan = _text(_two_slices(n)).plan
        assert plan["low"] >= 70 - n - 1  # no more "~1 more" at 59
        assert n + plan["low"] <= first <= n + plan["high"]


def test_planner_never_promises_a_pass_before_it_happens():
    first = _first_pass(3, 300)
    for n in range(3, first, 9):  # feed conversations in, one checkpoint every 9
        plan = _text(_two_slices(n)).plan
        assert plan is not None and n + plan["low"] <= first <= n + plan["high"]


def test_planner_says_nothing_at_or_above_alpha_and_flags_the_margin():
    assert _text(_two_slices(80, bad_every=10)).plan is None  # rate 0.1 >= alpha 0.05
    near = _text(_two_slices(200, bad_every=25)).plan  # rate 0.04, close to 0.05
    assert near and near["near_margin"] and "not a promise" in plan_text(near)
    assert not _text(_two_slices(40)).plan["near_margin"]  # rate 0 is not noisy


def test_planner_is_deterministic_and_resamples_whole_conversations():
    # per-conversation counts (2 wrong calls in one conversation) are resampled as units
    e, a = np.array([0, 2, 0, 0]), np.array([2, 2, 2, 2])
    first = more_sessions_needed(e, a, 0.3, 2, 0.05)
    assert first is not None and first == more_sessions_needed(e, a, 0.3, 2, 0.05)


def _one_slice(wrong: int, n: int):
    keys = ["s"] * n
    sess = [f"c{i}" for i in range(n)]
    return report_card(keys, sess, [i < wrong for i in range(n)], cap=1, plan=False)


def test_more_conversations_will_not_help_only_when_significantly_above_alpha():
    thin = _one_slice(1, 5).render()  # observed 20%, but the true rate may well be under 5%
    assert "will not help" not in thin and "too few to show it is above" in thin
    clear = _one_slice(20, 20).render()  # 20 of 20: the lower bound is far above 5%
    assert "will not help" in clear


def test_one_bad_session_of_ten_at_cap_10_stays_undecided():
    # 10 sessions x 10 calls; one session disagrees on all 10 calls. Per call that is 10%, and a
    # call-level bound would say "significantly above 5%"; per session it is 1 of 10, which is not
    # evidence that more conversations cannot help.
    keys, sess, wrong = [], [], []
    for s in range(10):
        for _ in range(10):
            keys.append("s")
            sess.append(f"c{s}")
            wrong.append(s == 0)
    card = report_card(keys, sess, wrong, cap=10, plan=False)
    assert not card.slices[0].above_alpha
    assert "will not help" not in card.render() and "cannot be decided" in card.render()


def test_render_counts_capped_calls_and_never_shows_detect_above_one():
    head, row = _one_slice(1, 1).render().splitlines()[:2]
    assert "counted" in head
    assert row.split()[7] == "-"  # one conversation: no rate can be ruled out
