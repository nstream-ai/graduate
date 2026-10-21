import numpy as np
import pytest

from graduate.certify.card import choose_caps, report_card, within_cap


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
        false += report_card(k, s, w, alpha=0.05, cap=2).slices[0].certified
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
