"""Tests for scripts/benchmark_routing.py.

Both routers are replaced by small fakes, so these tests never download the
embedding model, never call a provider and never need an API key. A few tests
read the real data/routing_benchmark.jsonl file to check its structure.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.routing import LONG_PROMPT_CHARS
from app.schemas import MAX_PROMPT_LENGTH, RouteResponse, SemanticRouteResponse
from app.semantic import CATEGORY_EXAMPLES, SemanticModelError
from scripts import benchmark_routing as bench

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_DATASET = PROJECT_ROOT / "data" / "routing_benchmark.jsonl"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_record(id="t-01", prompt="A test prompt.", category="coding", tier="cheap", **extra):
    record = {"id": id, "prompt": prompt, "expected_category": category,
              "expected_tier": tier, "label_note": "because"}
    record.update(extra)
    return record


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


# A tiny scenario whose metrics were worked out by hand.
#   id  expected (category, tier)   rule says          semantic says (similarity)
SCENARIO = [
    ("p1", "coding", "cheap",          ("coding", "cheap"),          ("coding", "cheap", 0.9)),
    ("p2", "coding", "strong",         ("general", "cheap"),         ("coding", "cheap", 0.8)),
    ("p3", "reasoning", "strong",      ("reasoning", "strong"),      ("reasoning", "strong", 0.7)),
    ("p4", "reasoning", "cheap",       ("reasoning", "strong"),      ("general", "strong", 0.3)),
    ("p5", "general", "cheap",         ("general", "cheap"),         ("general", "cheap", 0.6)),
    ("p6", "summarization", "cheap",   ("general", "cheap"),         ("summarization", "cheap", 0.5)),
]
SCENARIO_RECORDS = [make_record(id=i, prompt=i, category=c, tier=t) for i, c, t, _, _ in SCENARIO]


def fake_rule_router(prompt):
    row = next(r for r in SCENARIO if r[0] == prompt)
    return RouteResponse(task_category=row[3][0], model_tier=row[3][1], reason=f"rule:{prompt}")


def fake_semantic_router(prompt):
    row = next(r for r in SCENARIO if r[0] == prompt)
    category, tier, sim = row[4]
    return SemanticRouteResponse(task_category=category, model_tier=tier,
                                 similarity_score=sim, reason=f"semantic:{prompt}")


@pytest.fixture
def results():
    return bench.run_benchmark(SCENARIO_RECORDS, fake_rule_router, fake_semantic_router)


def render(results, **kwargs):
    defaults = dict(dataset_label="data/test.jsonl", dataset_sha256="abcdef1234567890",
                    generated_at="2026-01-01 00:00 UTC")
    defaults.update(kwargs)
    return bench.render_report(results, **defaults)


# --------------------------------------------------------------------------
# dataset validation
# --------------------------------------------------------------------------

def test_load_valid_dataset_strips_prompt_and_defaults_tags(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record(prompt="  hello there  ")])
    [record] = bench.load_dataset(path)
    assert record["prompt"] == "hello there"
    assert record["tags"] == []


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(make_record("a")) + "\n\n   \n" + json.dumps(make_record("b")) + "\n", encoding="utf-8")
    assert [r["id"] for r in bench.load_dataset(path)] == ["a", "b"]


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(bench.DatasetError, match="not found"):
        bench.load_dataset(tmp_path / "nope.jsonl")


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(bench.DatasetError, match="no records"):
        bench.load_dataset(path)


def test_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(make_record("a")) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(bench.DatasetError, match="line 2: invalid JSON"):
        bench.load_dataset(path)


def test_non_object_line_is_rejected(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(bench.DatasetError, match="JSON object"):
        bench.load_dataset(path)


@pytest.mark.parametrize("missing", list(bench.REQUIRED_FIELDS))
def test_each_required_field_is_enforced(tmp_path, missing):
    record = make_record()
    del record[missing]
    path = write_jsonl(tmp_path / "d.jsonl", [record])
    with pytest.raises(bench.DatasetError, match=f"missing field.*{missing}"):
        bench.load_dataset(path)


def test_unknown_field_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record(colour="red")])
    with pytest.raises(bench.DatasetError, match="unknown field.*colour"):
        bench.load_dataset(path)


def test_invalid_category_and_tier_are_rejected(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record("a", category="math"), make_record("b", tier="medium")])
    with pytest.raises(bench.DatasetError) as info:
        bench.load_dataset(path)
    message = str(info.value)
    assert "expected_category 'math'" in message
    assert "expected_tier 'medium'" in message


@pytest.mark.parametrize("prompt", ["", "   ", None, 5])
def test_empty_or_non_string_prompt_is_rejected(tmp_path, prompt):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record(prompt=prompt)])
    with pytest.raises(bench.DatasetError, match="'prompt' must be a non-empty string"):
        bench.load_dataset(path)


def test_prompt_longer_than_api_limit_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record(prompt="a" * (MAX_PROMPT_LENGTH + 1))])
    with pytest.raises(bench.DatasetError, match="API limit"):
        bench.load_dataset(path)


def test_duplicate_ids_are_rejected(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record("same"), make_record("same", prompt="other")])
    with pytest.raises(bench.DatasetError, match="duplicate id 'same'"):
        bench.load_dataset(path)


@pytest.mark.parametrize("tags", ["simple", ["nonsense-tag"], [1]])
def test_bad_tags_are_rejected(tmp_path, tags):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record(tags=tags)])
    with pytest.raises(bench.DatasetError, match="tag"):
        bench.load_dataset(path)


def test_every_problem_is_reported_not_just_the_first(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [make_record("a", category="x"), make_record("b", tier="y")])
    with pytest.raises(bench.DatasetError) as info:
        bench.load_dataset(path)
    assert "expected_category" in str(info.value) and "expected_tier" in str(info.value)


# --- leakage ---------------------------------------------------------------

def test_exact_copy_of_router_example_is_detected_ignoring_case_and_punctuation():
    example = CATEGORY_EXAMPLES["coding"][0]
    record = make_record(prompt=example.upper().replace(".", "!!"))
    exact, near = bench.find_leakage([record])
    assert [e["id"] for e in exact] == ["t-01"]


def test_check_leakage_raises_on_exact_copy():
    record = make_record(prompt=CATEGORY_EXAMPLES["general"][0])
    with pytest.raises(bench.DatasetError, match="must not reuse"):
        bench.check_leakage([record])


def test_near_duplicate_is_a_warning_not_an_error():
    almost = CATEGORY_EXAMPLES["summarization"][0] + " Please."
    near = bench.check_leakage([make_record(prompt=almost)])
    assert len(near) == 1 and near[0]["overlap"] >= bench.NEAR_DUPLICATE_THRESHOLD


def test_unrelated_prompt_has_no_leakage():
    assert bench.find_leakage([make_record(prompt="Describe how glaciers carve valleys.")]) == ([], [])


# --- the shipped dataset -----------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    return bench.load_dataset(SHIPPED_DATASET)


def test_shipped_dataset_has_60_prompts_12_per_category(shipped):
    assert len(shipped) == 60
    counts = {c: sum(r["expected_category"] == c for r in shipped) for c in bench.CATEGORIES}
    assert counts == {c: 12 for c in bench.CATEGORIES}


def test_shipped_dataset_has_both_tiers_and_unique_ids(shipped):
    assert {r["expected_tier"] for r in shipped} == {"cheap", "strong"}
    assert len({r["id"] for r in shipped}) == 60


def test_shipped_dataset_does_not_reuse_semantic_router_examples(shipped):
    exact, near = bench.find_leakage(shipped)
    assert exact == []
    assert near == []


def test_shipped_dataset_covers_the_required_prompt_types(shipped):
    tags = {t for r in shipped for t in r["tags"]}
    for required in ("simple", "difficult", "ambiguous", "short-but-difficult", "misleading-keyword"):
        assert required in tags


def test_shipped_dataset_exercises_the_length_heuristic(shipped):
    long_simple = [r for r in shipped if "long-but-simple" in r["tags"]]
    assert long_simple and all(len(r["prompt"]) >= LONG_PROMPT_CHARS for r in long_simple)
    assert all(r["expected_tier"] == "cheap" for r in long_simple)


def test_every_shipped_record_has_a_label_note(shipped):
    assert all(len(r["label_note"]) > 10 for r in shipped)


def test_fingerprint_ignores_windows_line_endings(tmp_path):
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    crlf.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
    assert bench.dataset_fingerprint(lf) == bench.dataset_fingerprint(crlf)


# --------------------------------------------------------------------------
# metrics (hand-computed values)
# --------------------------------------------------------------------------

def test_accuracy():
    assert bench.accuracy(["a", "b", "c", "d"], ["a", "b", "x", "x"]) == 0.5
    assert bench.accuracy([], []) == 0.0


def test_accuracy_rejects_length_mismatch():
    with pytest.raises(ValueError):
        bench.accuracy(["a"], ["a", "b"])


def test_wilson_interval_known_value_and_bounds():
    low, high = bench.wilson_interval(45, 60)
    assert low == pytest.approx(0.628, abs=0.002)
    assert high == pytest.approx(0.842, abs=0.002)
    assert bench.wilson_interval(0, 0) == (0.0, 0.0)
    assert bench.wilson_interval(60, 60)[1] == pytest.approx(1.0)
    assert bench.wilson_interval(0, 60)[0] == 0.0


def test_confusion_matrix_counts():
    labels = ["a", "b"]
    matrix = bench.confusion_matrix(["a", "a", "b", "b", "b"], ["a", "b", "b", "b", "a"], labels)
    assert matrix == {"a": {"a": 1, "b": 1}, "b": {"a": 1, "b": 2}}


def test_confusion_matrix_rejects_unknown_label():
    with pytest.raises(ValueError):
        bench.confusion_matrix(["a"], ["zzz"], ["a", "b"])


def test_confusion_matrix_rows_sum_to_support(results):
    expected = [r.record["expected_category"] for r in results]
    predicted = [r.rule.category for r in results]
    matrix = bench.confusion_matrix(expected, predicted, bench.CATEGORIES)
    assert sum(sum(row.values()) for row in matrix.values()) == len(results)
    assert matrix["coding"] == {"coding": 1, "reasoning": 0, "summarization": 0, "creative_writing": 0, "general": 1}
    assert matrix["summarization"]["general"] == 1


def test_per_class_metrics_rule_router(results):
    expected = [r.record["expected_category"] for r in results]
    metrics = bench.per_class_metrics(expected, [r.rule.category for r in results], bench.CATEGORIES)
    assert metrics["coding"].precision == pytest.approx(1.0)
    assert metrics["coding"].recall == pytest.approx(0.5)
    assert metrics["coding"].f1 == pytest.approx(2 / 3)
    assert metrics["reasoning"].f1 == pytest.approx(1.0)
    assert metrics["general"].precision == pytest.approx(1 / 3)
    assert metrics["general"].recall == pytest.approx(1.0)
    assert metrics["general"].f1 == pytest.approx(0.5)
    assert (metrics["summarization"].precision, metrics["summarization"].recall, metrics["summarization"].f1) == (0.0, 0.0, 0.0)
    assert metrics["general"].support == 1 and metrics["general"].predicted == 3


def test_zero_division_gives_zero_not_an_error():
    metrics = bench.per_class_metrics(["a", "a"], ["a", "a"], ["a", "b"])
    assert metrics["b"].precision == 0.0 and metrics["b"].recall == 0.0 and metrics["b"].f1 == 0.0


def test_macro_f1_ignores_labels_that_never_occur(results):
    expected = [r.record["expected_category"] for r in results]
    metrics = bench.per_class_metrics(expected, [r.rule.category for r in results], bench.CATEGORIES)
    # creative_writing never occurs or is predicted, so only 4 labels count: (2/3 + 1 + 0.5 + 0) / 4
    assert bench.macro_f1(metrics) == pytest.approx((2 / 3 + 1 + 0.5 + 0) / 4)


def test_macro_f1_extremes():
    perfect = bench.per_class_metrics(["a", "b"], ["a", "b"], ["a", "b"])
    wrong = bench.per_class_metrics(["a", "b"], ["b", "a"], ["a", "b"])
    assert bench.macro_f1(perfect) == 1.0
    assert bench.macro_f1(wrong) == 0.0


def test_tier_accuracy_and_error_types(results):
    expected = [r.record["expected_tier"] for r in results]
    rule = [r.rule.tier for r in results]
    assert bench.accuracy(expected, rule) == pytest.approx(4 / 6)
    assert bench.tier_error_breakdown(expected, rule) == {"over_routed": 1, "under_routed": 1}
    assert bench.always_cheap_accuracy(expected) == pytest.approx(4 / 6)


def test_tier_is_scored_per_router_against_expected_tier(results):
    assert sum(r.rule_tier_correct for r in results) == 4
    assert sum(r.semantic_tier_correct for r in results) == 4
    assert [r.record["id"] for r in results if not r.rule_tier_correct] == ["p2", "p4"]


def test_category_disagreements_and_who_was_right(results):
    assert [r.record["id"] for r in bench.category_disagreements(results)] == ["p2", "p4", "p6"]
    assert bench.disagreement_breakdown(results) == {
        "total": 3, "rule_only_correct": 1, "semantic_only_correct": 2, "both_wrong": 0}


def test_agreeing_routers_have_zero_disagreements():
    same = bench.run_benchmark([SCENARIO_RECORDS[0]], fake_rule_router, fake_semantic_router)
    assert bench.disagreement_breakdown(same)["total"] == 0


def test_low_similarity_selection_is_strict_and_sorted(results):
    assert [r.record["id"] for r in bench.low_similarity_results(results, 0.4)] == ["p4"]
    assert [r.record["id"] for r in bench.low_similarity_results(results, 0.8)] == ["p4", "p6", "p5", "p3"]
    assert bench.low_similarity_results(results, 0.1) == []


def test_mean_similarity_by_correctness(results):
    right, wrong = bench.mean_similarity_by_correctness(results)
    assert right == pytest.approx(0.7)
    assert wrong == pytest.approx(0.3)


# --------------------------------------------------------------------------
# running the routers
# --------------------------------------------------------------------------

def test_run_benchmark_collects_both_predictions_and_flags(results):
    assert [r.record["id"] for r in results] == ["p1", "p2", "p3", "p4", "p5", "p6"]
    p2 = results[1]
    assert (p2.rule.category, p2.rule.tier, p2.rule.similarity) == ("general", "cheap", None)
    assert (p2.semantic.category, p2.semantic.similarity) == ("coding", 0.8)
    assert p2.rule_category_correct is False and p2.semantic_category_correct is True
    assert p2.categories_agree is False


def test_semantic_model_failure_becomes_a_clear_benchmark_error():
    def broken(prompt):
        raise SemanticModelError("Could not load embedding model 'x'.")

    with pytest.raises(bench.BenchmarkError, match="semantic router could not run"):
        bench.run_benchmark(SCENARIO_RECORDS, fake_rule_router, broken)


def test_unexpected_router_errors_name_the_prompt_id_but_not_the_message():
    def buggy(prompt):
        raise ValueError("secret detail")

    with pytest.raises(bench.BenchmarkError) as info:
        bench.run_benchmark(SCENARIO_RECORDS, buggy, fake_semantic_router)
    assert "p1" in str(info.value) and "ValueError" in str(info.value)
    assert "secret detail" not in str(info.value)


# --------------------------------------------------------------------------
# report text
# --------------------------------------------------------------------------

def test_report_contains_every_section_and_the_caveats(results):
    text = render(results)
    for heading in ("1. DATASET", "2. TASK-CATEGORY CLASSIFICATION", "3. MODEL-TIER DECISIONS",
                    "4. ROUTER DISAGREEMENT", "5. SEMANTIC SIMILARITY", "6. PROMPTS TO INSPECT MANUALLY"):
        assert heading in text
    assert "human-designed expectations" in text
    assert "not production evidence" in text
    assert "abcdef123456" in text  # dataset fingerprint (first 12 characters)
    assert "none (routing decisions only" in text  # no LLM calls


def test_report_numbers_match_the_hand_computed_values(results):
    text = render(results)
    assert "4/6" in text and "66.7%" in text    # rule-based category accuracy
    assert "5/6" in text and "83.3%" in text    # semantic category accuracy
    assert "rule-based 0.542" in text           # macro-F1 (0.6667+1+0.5+0)/4
    assert "over-routed (expected cheap -> strong) = 1" in text
    assert "under-routed (expected strong -> cheap) = 1" in text
    assert "chose different tiers: 0" in text
    assert "Different category chosen: 3 of 6" in text
    assert "only semantic matched the label:   2" in text


def test_report_shows_confusion_matrices_for_both_routers(results):
    text = render(results)
    assert text.count("expected \\ predicted") >= 3  # rule, semantic, tier
    for label in bench.CATEGORIES:
        assert label in text


def test_report_lists_misclassified_prompts_and_low_similarity(results):
    text = render(results, low_threshold=0.4)
    rule_part = text.split("Category errors - rule-based:")[1].split("Category errors - semantic:")[0]
    semantic_part = text.split("Category errors - semantic:")[1].split("Tier errors - rule-based:")[0]
    assert "[p2]" in rule_part and "[p6]" in rule_part and "[p1]" not in rule_part
    assert "[p4]" in semantic_part and "[p2]" not in semantic_part
    low_part = text.split("reporting threshold")[1].split("Mean similarity")[0]
    assert "[p4] sim=0.300" in low_part and "[p5]" not in low_part


def test_report_says_none_when_there_are_no_errors():
    perfect = [make_record("p1", "q", "coding", "cheap")]
    ok = bench.run_benchmark(
        perfect,
        lambda p: RouteResponse(task_category="coding", model_tier="cheap", reason="r"),
        lambda p: SemanticRouteResponse(task_category="coding", model_tier="cheap", similarity_score=0.9, reason="r"),
    )
    text = render(ok)
    assert text.split("Category errors - rule-based:")[1].lstrip().startswith("none")


def test_report_truncates_long_lists_but_points_to_the_csv(results):
    text = render(results, max_listed=1)
    assert "more (see the CSV)" in text


def test_report_warns_about_near_duplicate_prompts(results):
    near = [{"id": "p1", "example": "some example", "overlap": 0.8, "category": "coding"}]
    assert "WARNING: 1 prompt(s) look very similar" in render(results, near_duplicates=near)


def test_report_needs_results():
    with pytest.raises(ValueError):
        render([])


# --------------------------------------------------------------------------
# CSV output
# --------------------------------------------------------------------------

def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_csv_creates_missing_folders_and_has_one_row_per_prompt(results, tmp_path):
    target = tmp_path / "deep" / "er" / "out.csv"
    bench.write_csv(results, target)
    rows = read_csv(target)
    assert len(rows) == 6
    assert list(rows[0].keys()) == bench.CSV_COLUMNS


def test_csv_contains_labels_predictions_scores_and_flags(results, tmp_path):
    target = bench.write_csv(results, tmp_path / "out.csv", low_threshold=0.4)
    rows = {r["id"]: r for r in read_csv(target)}
    p2 = rows["p2"]
    assert (p2["expected_category"], p2["expected_tier"]) == ("coding", "strong")
    assert (p2["rule_category"], p2["rule_tier"]) == ("general", "cheap")
    assert (p2["semantic_category"], p2["semantic_tier"]) == ("coding", "cheap")
    assert p2["semantic_similarity"] == "0.8000"
    assert p2["rule_category_correct"] == "false" and p2["semantic_category_correct"] == "true"
    assert p2["rule_tier_correct"] == "false" and p2["semantic_tier_correct"] == "false"
    assert p2["categories_agree"] == "false"
    assert p2["low_similarity"] == "false" and rows["p4"]["low_similarity"] == "true"
    assert p2["rule_reason"] == "rule:p2" and p2["semantic_reason"] == "semantic:p2"


def test_csv_round_trips_awkward_text_and_tags(tmp_path):
    record = make_record("odd", prompt='He said, "hi", then\nleft. Caf\u00e9 \u2013 ok', tags=["simple", "ambiguous"])
    res = bench.run_benchmark(
        [record],
        lambda p: RouteResponse(task_category="coding", model_tier="cheap", reason="r"),
        lambda p: SemanticRouteResponse(task_category="coding", model_tier="cheap", similarity_score=0.5, reason="r"),
    )
    [row] = read_csv(bench.write_csv(res, tmp_path / "o.csv"))
    assert row["prompt"] == record["prompt"]
    assert row["tags"] == "simple|ambiguous"


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def run_main(tmp_path, records=None, extra=(), semantic=fake_semantic_router, rule=fake_rule_router):
    dataset = write_jsonl(tmp_path / "in.jsonl", records if records is not None else SCENARIO_RECORDS)
    output = tmp_path / "reports" / "out.csv"
    code = bench.main(["--dataset", str(dataset), "--output", str(output), *extra], rule, semantic)
    return code, output


def test_defaults_point_at_the_documented_paths():
    args = bench.parse_args([])
    assert args.dataset == PROJECT_ROOT / "data" / "routing_benchmark.jsonl"
    assert args.output == PROJECT_ROOT / "reports" / "routing_benchmark_results.csv"


def test_main_success_prints_summary_and_writes_csv(tmp_path, capsys):
    code, output = run_main(tmp_path)
    captured = capsys.readouterr()
    assert code == 0
    assert "ROUTEWISE ROUTING BENCHMARK" in captured.out
    assert "Per-prompt CSV written to" in captured.out
    assert len(read_csv(output)) == 6


def test_main_invalid_dataset_exits_2_and_writes_no_csv(tmp_path, capsys):
    code, output = run_main(tmp_path, records=[make_record(category="nonsense")])
    assert code == 2
    assert "invalid dataset" in capsys.readouterr().err
    assert not output.exists()


def test_main_rejects_dataset_that_copies_router_examples(tmp_path, capsys):
    code, _ = run_main(tmp_path, records=[make_record(prompt=CATEGORY_EXAMPLES["coding"][1])])
    assert code == 2
    assert "must not reuse" in capsys.readouterr().err


def test_main_missing_dataset_exits_2(tmp_path, capsys):
    code = bench.main(["--dataset", str(tmp_path / "missing.jsonl"), "--output", str(tmp_path / "o.csv")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_main_semantic_model_failure_exits_3_with_clear_message(tmp_path, capsys):
    def broken(prompt):
        raise SemanticModelError("Could not load embedding model 'x'.")

    code, output = run_main(tmp_path, semantic=broken)
    assert code == 3
    assert "semantic router could not run" in capsys.readouterr().err
    assert not output.exists()


def test_main_unwritable_output_exits_4(tmp_path, capsys):
    dataset = write_jsonl(tmp_path / "in.jsonl", SCENARIO_RECORDS)
    blocker = tmp_path / "is_a_folder"
    blocker.mkdir()
    code = bench.main(["--dataset", str(dataset), "--output", str(blocker)], fake_rule_router, fake_semantic_router)
    assert code == 4
    assert "could not write the CSV" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [["--low-similarity-threshold", "0"], ["--low-similarity-threshold", "1.5"], ["--max-listed", "0"]])
def test_bad_command_line_options_are_rejected(bad):
    with pytest.raises(SystemExit) as info:
        bench.parse_args(bad)
    assert info.value.code == 2


def test_full_pipeline_on_shipped_dataset_with_real_rule_router_and_fake_semantic(tmp_path, capsys):
    def semantic(prompt):
        return SemanticRouteResponse(task_category="general", model_tier="cheap", similarity_score=0.5, reason="fake")

    output = tmp_path / "shipped.csv"
    code = bench.main(["--output", str(output)], None, semantic)
    assert code == 0
    assert len(read_csv(output)) == 60
    assert "60 prompts" in capsys.readouterr().out


# --------------------------------------------------------------------------
# safety: no LLM, no heavy dependencies
# --------------------------------------------------------------------------

def test_importing_the_script_loads_no_llm_client_embedding_model_or_dataframe_library():
    code = (
        "import sys; import scripts.benchmark_routing; "
        "bad = [m for m in ('litellm', 'sentence_transformers', 'pandas', 'sklearn') if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_benchmark_never_touches_the_observability_database(tmp_path, monkeypatch, capsys):
    db = tmp_path / "must_not_exist.db"
    monkeypatch.setenv("ROUTEWISE_DB_PATH", str(db))
    run_main(tmp_path)
    assert not db.exists()
