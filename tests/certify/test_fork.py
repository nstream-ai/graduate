"""Claim B by effect (graduate.certify.fork): MDE, verdicts, session bound, reference floor."""

from __future__ import annotations

import json
import math
from pathlib import Path

from graduate.certify.fork import Z_POWER, fork_card, slice_stats
from graduate.cli import main


def _pairs(n, worse=0, noisy=0, tasks=30, sl="s", t1=True):
    out = []
    for i in range(n):
        s = not (i < worse) if t1 else t1
        t2 = t1 if i < n - noisy else not t1
        out.append({"state": f"{sl}{i}", "task_id": f"t{i % tasks}", "slice": sl,
                    "proposed_slice": sl, "T1": t1, "T2": t2, "S": s})  # fmt: skip
    return out


def test_detectable_harm_follows_the_pre_registered_rule() -> None:
    st = slice_stats(_pairs(100, noisy=25), delta=0.10)
    assert math.isclose(st["detectable_harm"], Z_POWER * math.sqrt(0.25 / 100))
    st0 = slice_stats(_pairs(100), delta=0.10)  # no noise: the zero-event floor 3 / n
    assert math.isclose(st0["detectable_harm"], 3 / 100)


def test_verdicts_cannot_be_decided_is_not_a_failure() -> None:
    pairs = (_pairs(200, sl="clean", tasks=100) + _pairs(25, noisy=8, sl="thin", tasks=25)
             + _pairs(80, worse=60, sl="harm") + _pairs(80, sl="dead", t1=False))  # fmt: skip
    card = fork_card(pairs, session_margin=0.10, cap=1, min_states=20, serving={"model": "ref"})
    t = card["by_proposed_slice"]
    assert t["clean"]["verdict"] == "certified"
    assert t["thin"]["verdict"].startswith("cannot be decided with this evidence")
    assert t["harm"]["verdict"].startswith("excluded by evidence")
    assert t["dead"]["verdict"].startswith("vacuous")
    assert t["clean"]["session_harm_upper"] == min(1.0, 1 * t["clean"]["harm_upper"])
    assert card["serving"] == {"model": "ref"} and "h_T" in card["reference_floor"]


def test_session_bound_composes_over_the_cap() -> None:
    card = fork_card(_pairs(300, worse=3), session_margin=0.10, cap=2, min_states=20)
    t = card["by_proposed_slice"]["s"]
    assert card["delta_turn"] == 0.05
    assert math.isclose(t["session_harm_upper"], min(1.0, 2 * t["harm_upper"]))


def test_claim_b_cli(tmp_path: Path) -> None:
    rows = []
    for i in range(120):
        for b in ("T1", "T2", "S"):
            rows.append({"state": f"x{i}", "branch": b, "task_id": f"t{i % 60}",
                         "slice": "retail/tool/get_order/4-10",
                         "proposed_slice": "retail/tool/get_order/4-10",
                         "termination": "user_stop", "success": i % 3 != 0})  # fmt: skip
    f = tmp_path / "forks.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "cb"
    main(["claim-b", str(f), "--out", str(out), "--serving", '{"temperature": 0}'])
    card = json.loads((out / "claim_b.json").read_text())
    assert card["routing"]["certified"] == ["retail/tool/get_order"]  # positions pooled
    text = (out / "claim_b.md").read_text()
    assert "detectable harm" in text and "union bound" in text and "reset to a logged state" in text


def test_cap_max_informative_is_the_margin_over_the_per_turn_bound():
    import math

    from graduate.certify.fork import fork_card

    pairs = [
        {"state": str(i), "task_id": f"t{i % 40}", "slice": "a", "proposed_slice": "a",
         "T1": i % 2 == 0, "T2": i % 2 == 0, "S": i % 2 == 0 if i % 25 else i % 2 == 1}
        for i in range(400)
    ]  # fmt: skip
    card = fork_card(pairs, session_margin=0.10, cap=1)
    # 40 tasks with some harm: nothing certifies, so no cap is informative
    assert not card["by_proposed_slice"]["a"]["certified"]
    assert card["cap_max_informative"] is None and "no certified slice" in card["composition"]
    clean = [
        {"state": str(i), "task_id": f"t{i}", "slice": "b", "proposed_slice": "b",
         "T1": True, "T2": True, "S": True}
        for i in range(200)
    ]  # fmt: skip
    card = fork_card(pairs + clean, session_margin=0.10, cap=1)
    upper = card["by_proposed_slice"]["b"]["harm_upper"]  # from the certified slice only
    assert card["by_proposed_slice"]["b"]["certified"]
    assert card["cap_max_informative"] == math.floor(0.10 / upper)
    assert f"informative up to cap {card['cap_max_informative']}" in card["composition"]
