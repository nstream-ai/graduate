"""Session-level claim B (graduate.certify.session) and `graduate compare`."""

from __future__ import annotations

import json
from pathlib import Path

from graduate.certify.session import compare
from graduate.cli import main


def test_non_inferior_when_the_candidate_matches() -> None:
    ref = {f"t{i}": [1.0, 0.0] for i in range(200)}
    r = compare(ref, dict(ref), margin=0.10)
    assert r["regret"] == 0 and r["verdict"] == "non-inferior"
    assert r["tasks_worse"] == r["tasks_better"] == 0


def test_a_clearly_worse_candidate_is_not_non_inferior() -> None:
    ref = {f"t{i}": [1.0] for i in range(200)}
    cand = {f"t{i}": [0.0 if i % 2 else 1.0] for i in range(200)}  # 50 points worse
    r = compare(ref, cand, margin=0.10)
    assert abs(r["regret"] - 0.5) < 1e-12 and r["verdict"] == "not non-inferior"


def test_too_few_tasks_cannot_be_decided() -> None:
    ref = {f"t{i}": [1.0 if i % 2 else 0.0] for i in range(12)}
    cand = {f"t{i}": [0.0 if i % 2 else 1.0] for i in range(12)}  # flips everywhere, mean 0
    r = compare(ref, cand, margin=0.10)
    assert r["detectable_regret"] > 0.10
    assert r["verdict"] == "cannot be decided with this evidence"


def test_compare_cli_reads_jsonl_rows(tmp_path: Path) -> None:
    ref = tmp_path / "ref.jsonl"
    cand = tmp_path / "cand.jsonl"
    ref.write_text("\n".join(json.dumps({"task_id": i, "success": True}) for i in range(150)))
    cand.write_text("\n".join(json.dumps({"task_id": i, "success": True}) for i in range(150)))
    main(["compare", str(ref), str(cand), "--out", str(tmp_path / "c")])
    r = json.loads((tmp_path / "c" / "compare.json").read_text())
    assert r["tasks"] == 150 and r["verdict"] == "non-inferior"


def test_vacuous_when_the_reference_rarely_succeeds_and_both_bounds_are_named():
    from graduate.certify.session import compare

    ref = {f"t{i}": [1.0 if i < 3 else 0.0] for i in range(30)}  # reference succeeds 10%
    cand = {f"t{i}": [0.0] for i in range(30)}
    r = compare(ref, cand, margin=0.10)
    assert not r["non_vacuous"] and r["verdict"].startswith("vacuous")
    ok = compare({f"t{i}": [1.0] for i in range(30)}, {f"t{i}": [1.0] for i in range(30)})
    assert ok["non_vacuous"] and ok["binding"] in ("bootstrap", "exact")
    assert {"regret_upper_bootstrap", "regret_upper_exact"} <= set(ok)
