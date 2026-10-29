"""Routing benchmark: rule-based router vs semantic-plus-rules router.

What it does
    Loads a labelled JSONL dataset, sends every prompt through BOTH routers
    (app.routing.route_prompt and app.semantic.route_prompt_semantic), compares
    their decisions with the human labels, prints a summary and writes a
    per-prompt CSV.

What it does NOT do
    No LLM generation, no provider call, no API key, no paid service, no
    database writes. Only the local embedding model is used (semantic router).
    It measures ROUTING DECISIONS against labels, not answer quality or cost.

Run from the project root:
    python -m scripts.benchmark_routing
    python -m scripts.benchmark_routing --dataset data\\routing_benchmark.jsonl --output reports\\routing_benchmark_results.csv

Exit codes: 0 ok | 2 invalid dataset | 3 router / embedding-model failure | 4 CSV could not be written
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence, get_args

ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):  # started as "python scripts\benchmark_routing.py"
    sys.path.insert(0, str(ROOT))

from app.routing import route_prompt  # noqa: E402
from app.schemas import MAX_PROMPT_LENGTH, ModelTier, TaskCategory  # noqa: E402
from app.semantic import (  # noqa: E402
    CATEGORY_EXAMPLES,
    MIN_SIMILARITY,
    MODEL_NAME,
    TOP_K,
    SemanticModelError,
    route_prompt_semantic,
)

DEFAULT_DATASET = ROOT / "data" / "routing_benchmark.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "routing_benchmark_results.csv"

CATEGORIES: tuple[str, ...] = get_args(TaskCategory)
TIERS: tuple[str, ...] = get_args(ModelTier)

REQUIRED_FIELDS = ("id", "prompt", "expected_category", "expected_tier", "label_note")
OPTIONAL_FIELDS = ("tags",)
ALLOWED_TAGS = (
    "simple", "moderate", "difficult", "ambiguous", "short-but-difficult",
    "misleading-keyword", "keyword-free", "long-but-simple",
)

# A benchmark prompt this similar (word overlap) to a semantic-router example is flagged.
NEAR_DUPLICATE_THRESHOLD = 0.7

# Reporting threshold for "low" semantic similarity. This is NOT the router's own
# fallback threshold (MIN_SIMILARITY) and is not tuned; change it with a CLI option.
DEFAULT_LOW_SIMILARITY = 0.40


class DatasetError(ValueError):
    """The dataset file is missing, unreadable or invalid."""


class BenchmarkError(RuntimeError):
    """A router (or the embedding model) failed while running the benchmark."""


# ---------------------------------------------------------------------------
# 1. Dataset loading and validation
# ---------------------------------------------------------------------------

def validate_record(record: object, where: str) -> list[str]:
    """Return a list of problems with one record (empty list = valid)."""
    if not isinstance(record, dict):
        return [f"{where}: each line must be a JSON object"]

    errors = []
    label = f"{where} (id={record.get('id')!r})" if isinstance(record.get("id"), str) else where

    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        errors.append(f"{label}: missing field(s): {', '.join(missing)}")
    unknown = sorted(set(record) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        errors.append(f"{label}: unknown field(s): {', '.join(unknown)}")

    for name in ("id", "prompt", "label_note"):
        if name in record and (not isinstance(record[name], str) or not record[name].strip()):
            errors.append(f"{label}: '{name}' must be a non-empty string")

    prompt = record.get("prompt")
    if isinstance(prompt, str) and len(prompt.strip()) > MAX_PROMPT_LENGTH:
        errors.append(f"{label}: prompt is longer than the API limit of {MAX_PROMPT_LENGTH} characters")

    if "expected_category" in record and record["expected_category"] not in CATEGORIES:
        errors.append(
            f"{label}: expected_category {record['expected_category']!r} is not one of {list(CATEGORIES)}"
        )
    if "expected_tier" in record and record["expected_tier"] not in TIERS:
        errors.append(f"{label}: expected_tier {record['expected_tier']!r} is not one of {list(TIERS)}")

    if "tags" in record:
        tags = record["tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            errors.append(f"{label}: 'tags' must be a list of strings")
        else:
            bad = [t for t in tags if t not in ALLOWED_TAGS]
            if bad:
                errors.append(f"{label}: unknown tag(s) {bad}; allowed: {list(ALLOWED_TAGS)}")
    return errors


def load_dataset(path: Path | str) -> list[dict]:
    """Read and validate a JSONL dataset. Raises DatasetError listing every problem found."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not read dataset {path}: {type(exc).__name__}") from exc

    records, errors = [], []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue  # blank lines are allowed
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
            continue
        problems = validate_record(record, f"line {line_number}")
        if problems:
            errors.extend(problems)
            continue
        record = dict(record)
        record["prompt"] = record["prompt"].strip()  # the API strips whitespace too
        record["tags"] = list(record.get("tags", []))
        records.append(record)

    seen: dict[str, int] = {}
    for record in records:
        seen[record["id"]] = seen.get(record["id"], 0) + 1
    errors.extend(f"duplicate id {rid!r} appears {n} times" for rid, n in seen.items() if n > 1)

    if not records and not errors:
        errors.append("the dataset contains no records")
    if errors:
        shown = errors[:20]
        if len(errors) > 20:
            shown.append(f"... and {len(errors) - 20} more problem(s)")
        raise DatasetError("\n".join(f"  - {e}" for e in shown))
    return records


def dataset_fingerprint(path: Path | str) -> str:
    """SHA-256 of the dataset with Windows line endings normalised, so the same content
    gives the same fingerprint on every machine. Shown in the report to prove which prompt set was used."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def find_leakage(
    records: Sequence[dict],
    examples: dict[str, Sequence[str]] | None = None,
    near_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[list[dict], list[dict]]:
    """Compare benchmark prompts with the semantic router's own examples.

    Returns (exact, near): exact = same words in the same order (ignoring case and
    punctuation); near = word-overlap (Jaccard) >= near_threshold.
    """
    examples = CATEGORY_EXAMPLES if examples is None else examples
    exact, near = [], []
    for record in records:
        prompt_tokens = _tokens(record["prompt"])
        prompt_norm = " ".join(re.findall(r"[a-z0-9]+", record["prompt"].lower()))
        for category, texts in examples.items():
            for example in texts:
                example_norm = " ".join(re.findall(r"[a-z0-9]+", example.lower()))
                if prompt_norm == example_norm:
                    exact.append({"id": record["id"], "category": category, "example": example})
                    continue
                example_tokens = _tokens(example)
                union = prompt_tokens | example_tokens
                overlap = len(prompt_tokens & example_tokens) / len(union) if union else 0.0
                if overlap >= near_threshold:
                    near.append({"id": record["id"], "category": category, "example": example,
                                 "overlap": round(overlap, 2)})
    return exact, near


def check_leakage(records: Sequence[dict]) -> list[dict]:
    """Raise DatasetError on exact copies of router examples; return near-duplicate warnings."""
    exact, near = find_leakage(records)
    if exact:
        lines = [f"  - {e['id']} is identical to a {e['category']} example: {e['example']!r}" for e in exact]
        raise DatasetError("benchmark prompts must not reuse the semantic router's examples:\n" + "\n".join(lines))
    return near


# ---------------------------------------------------------------------------
# 2. Running the routers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prediction:
    category: str
    tier: str
    similarity: float | None  # only the semantic router has one
    reason: str


def to_prediction(decision) -> Prediction:
    return Prediction(
        category=decision.task_category,
        tier=decision.model_tier,
        similarity=getattr(decision, "similarity_score", None),
        reason=decision.reason,
    )


@dataclass(frozen=True)
class PromptResult:
    record: dict
    rule: Prediction
    semantic: Prediction

    @property
    def rule_category_correct(self) -> bool:
        return self.rule.category == self.record["expected_category"]

    @property
    def rule_tier_correct(self) -> bool:
        return self.rule.tier == self.record["expected_tier"]

    @property
    def semantic_category_correct(self) -> bool:
        return self.semantic.category == self.record["expected_category"]

    @property
    def semantic_tier_correct(self) -> bool:
        return self.semantic.tier == self.record["expected_tier"]

    @property
    def categories_agree(self) -> bool:
        return self.rule.category == self.semantic.category


Router = Callable[[str], object]


def run_benchmark(
    records: Sequence[dict],
    rule_router: Router | None = None,
    semantic_router: Router | None = None,
) -> list[PromptResult]:
    """Run every prompt through both routers. The routers can be replaced (tests do this)."""
    rule_router = rule_router or route_prompt
    semantic_router = semantic_router or route_prompt_semantic
    results = []
    for record in records:
        try:
            rule = to_prediction(rule_router(record["prompt"]))
        except Exception as exc:
            raise BenchmarkError(f"Rule-based router failed on {record['id']}: {type(exc).__name__}") from exc
        try:
            semantic = to_prediction(semantic_router(record["prompt"]))
        except SemanticModelError as exc:
            raise BenchmarkError(
                f"The semantic router could not run: {exc}\n"
                "Check that sentence-transformers is installed (python -m pip install -r requirements.txt) "
                "and that the model can be downloaded on the first run (internet needed once)."
            ) from exc
        except Exception as exc:
            raise BenchmarkError(f"Semantic router failed on {record['id']}: {type(exc).__name__}") from exc
        results.append(PromptResult(record, rule, semantic))
    return results


# ---------------------------------------------------------------------------
# 3. Metrics (standard library only)
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Fraction of exact matches. Returns 0.0 for empty input."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return _safe_div(sum(t == p for t, p in zip(y_true, y_pred)), len(y_true))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion. Shows how uncertain a small sample is."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """matrix[true_label][predicted_label] = number of prompts. Rows = truth, columns = prediction."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t not in matrix or p not in matrix[t]:
            raise ValueError(f"label not in {list(labels)}: true={t!r}, predicted={p!r}")
        matrix[t][p] += 1
    return matrix


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int    # how many prompts truly belong to this class
    predicted: int  # how many prompts the router assigned to this class


def per_class_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, ClassMetrics]:
    """Precision, recall and F1 per label. A zero denominator gives 0.0 (the usual convention)."""
    matrix = confusion_matrix(y_true, y_pred, labels)
    metrics = {}
    for label in labels:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[t][label] for t in labels)
        precision = _safe_div(true_positive, predicted)
        recall = _safe_div(true_positive, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        metrics[label] = ClassMetrics(precision, recall, f1, support, predicted)
    return metrics


def macro_f1(metrics: dict[str, ClassMetrics]) -> float:
    """Unweighted mean F1 over labels that occur in the truth or in the predictions."""
    scores = [m.f1 for m in metrics.values() if m.support > 0 or m.predicted > 0]
    return _safe_div(sum(scores), len(scores))


def tier_error_breakdown(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, int]:
    """over_routed = expected cheap but sent to strong (wastes money);
    under_routed = expected strong but sent to cheap (risks quality)."""
    pairs = list(zip(y_true, y_pred))
    return {
        "over_routed": sum(t == "cheap" and p == "strong" for t, p in pairs),
        "under_routed": sum(t == "strong" and p == "cheap" for t, p in pairs),
    }


def always_cheap_accuracy(y_true: Sequence[str]) -> float:
    """Tier accuracy of a 'router' that always answers cheap. Any real router should beat this."""
    return _safe_div(sum(t == "cheap" for t in y_true), len(y_true))


def category_disagreements(results: Sequence[PromptResult]) -> list[PromptResult]:
    return [r for r in results if not r.categories_agree]


def disagreement_breakdown(results: Sequence[PromptResult]) -> dict[str, int]:
    """Among prompts where the routers pick different categories, who matched the label?"""
    disagreements = category_disagreements(results)
    return {
        "total": len(disagreements),
        "rule_only_correct": sum(r.rule_category_correct for r in disagreements),
        "semantic_only_correct": sum(r.semantic_category_correct for r in disagreements),
        "both_wrong": sum(not r.rule_category_correct and not r.semantic_category_correct for r in disagreements),
    }


def low_similarity_results(results: Sequence[PromptResult], threshold: float) -> list[PromptResult]:
    """Prompts whose semantic similarity is below `threshold`, lowest first."""
    low = [r for r in results if r.semantic.similarity is not None and r.semantic.similarity < threshold]
    return sorted(low, key=lambda r: r.semantic.similarity)


def mean_similarity_by_correctness(results: Sequence[PromptResult]) -> tuple[float | None, float | None]:
    """(mean similarity when the semantic category was right, mean when it was wrong)."""
    def mean(values):
        return sum(values) / len(values) if values else None

    scored = [r for r in results if r.semantic.similarity is not None]
    return (
        mean([r.semantic.similarity for r in scored if r.semantic_category_correct]),
        mean([r.semantic.similarity for r in scored if not r.semantic_category_correct]),
    )


# ---------------------------------------------------------------------------
# 4. Report text
# ---------------------------------------------------------------------------

def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _accuracy_line(name: str, correct: int, total: int) -> str:
    low, high = wilson_interval(correct, total)
    return f"  {name:<22} {correct:>3}/{total:<3} = {_pct(_safe_div(correct, total)):>6}   (95% CI {_pct(low)} to {_pct(high)})"


def _short(text: str, width: int = 88) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def format_confusion_matrix(matrix: dict[str, dict[str, int]], labels: Sequence[str]) -> str:
    left = max(len("expected \\ predicted"), *(len(label) for label in labels))
    widths = [max(len(label), 4) + 2 for label in labels]
    lines = ["  " + "expected \\ predicted".ljust(left) + "".join(l.rjust(w) for l, w in zip(labels, widths))]
    for true_label in labels:
        row = "".join(str(matrix[true_label][p]).rjust(w) for p, w in zip(labels, widths))
        lines.append("  " + true_label.ljust(left) + row)
    return "\n".join(lines)


def _format_class_table(metrics: dict[str, ClassMetrics], labels: Sequence[str]) -> str:
    left = max(len(label) for label in labels)
    lines = ["  " + "category".ljust(left) + "  precision   recall       F1  support  predicted"]
    for label in labels:
        m = metrics[label]
        lines.append(
            f"  {label.ljust(left)}  {m.precision:>9.3f}  {m.recall:>7.3f}  {m.f1:>7.3f}  {m.support:>7}  {m.predicted:>9}"
        )
    return "\n".join(lines)


def _list_items(lines: list[str], max_listed: int) -> list[str]:
    if len(lines) <= max_listed:
        return lines
    return lines[:max_listed] + [f"  ... {len(lines) - max_listed} more (see the CSV)"]


def render_report(
    results: Sequence[PromptResult],
    *,
    dataset_label: str,
    dataset_sha256: str,
    generated_at: str,
    low_threshold: float = DEFAULT_LOW_SIMILARITY,
    max_listed: int = 40,
    near_duplicates: Sequence[dict] = (),
) -> str:
    n = len(results)
    if n == 0:
        raise ValueError("cannot build a report from zero results")
    expected_cat = [r.record["expected_category"] for r in results]
    expected_tier = [r.record["expected_tier"] for r in results]
    rule_cat = [r.rule.category for r in results]
    sem_cat = [r.semantic.category for r in results]
    rule_tier = [r.rule.tier for r in results]
    sem_tier = [r.semantic.tier for r in results]

    out: list[str] = []
    bar = "=" * 78
    out += [bar, "ROUTEWISE ROUTING BENCHMARK", bar,
            f"Generated:      {generated_at}",
            f"Dataset:        {dataset_label}  ({n} prompts, sha256 {dataset_sha256[:12]})",
            f"Semantic model: {MODEL_NAME}  (TOP_K={TOP_K}, fallback below MIN_SIMILARITY={MIN_SIMILARITY})",
            "LLM calls:      none (routing decisions only; no answer quality, latency or cost is measured)",
            "",
            "NOTE: the expected labels are human-designed expectations and may need review.",
            "      Treat this as a small synthetic debugging benchmark, not production evidence.", ""]

    # --- composition
    out += ["1. DATASET", "-" * 78,
            "  expected categories: " + ", ".join(f"{c}={Counter(expected_cat)[c]}" for c in CATEGORIES),
            "  expected tiers:      " + ", ".join(f"{t}={Counter(expected_tier)[t]}" for t in TIERS)]
    tag_counts = Counter(tag for r in results for tag in r.record.get("tags", []))
    if tag_counts:
        out.append("  tags: " + ", ".join(f"{t}={c}" for t, c in sorted(tag_counts.items())))
    if near_duplicates:
        out.append(f"  WARNING: {len(near_duplicates)} prompt(s) look very similar to a semantic-router example:")
        out += [f"    {d['id']} ~ {d['example']!r} (word overlap {d['overlap']})" for d in near_duplicates]
    out.append("")

    # --- category
    rule_ok = sum(r.rule_category_correct for r in results)
    sem_ok = sum(r.semantic_category_correct for r in results)
    rule_m = per_class_metrics(expected_cat, rule_cat, CATEGORIES)
    sem_m = per_class_metrics(expected_cat, sem_cat, CATEGORIES)
    out += ["2. TASK-CATEGORY CLASSIFICATION", "-" * 78, "Accuracy (higher is better; a coin-flip-style guess among 5 classes is ~20%):",
            _accuracy_line("rule-based", rule_ok, n), _accuracy_line("semantic", sem_ok, n),
            f"  Macro-F1:  rule-based {macro_f1(rule_m):.3f}   semantic {macro_f1(sem_m):.3f}",
            "", "Rule-based: per-category precision / recall / F1", _format_class_table(rule_m, CATEGORIES),
            "", "Semantic: per-category precision / recall / F1", _format_class_table(sem_m, CATEGORIES),
            "", "Confusion matrix - rule-based (rows = expected, columns = predicted)",
            format_confusion_matrix(confusion_matrix(expected_cat, rule_cat, CATEGORIES), CATEGORIES),
            "", "Confusion matrix - semantic (rows = expected, columns = predicted)",
            format_confusion_matrix(confusion_matrix(expected_cat, sem_cat, CATEGORIES), CATEGORIES), ""]

    # --- tier
    out += ["3. MODEL-TIER DECISIONS (against the independently labelled expected_tier)", "-" * 78,
            _accuracy_line("rule-based tier", sum(r.rule_tier_correct for r in results), n),
            _accuracy_line("semantic tier", sum(r.semantic_tier_correct for r in results), n),
            _accuracy_line("always-cheap baseline", sum(t == "cheap" for t in expected_tier), n)]
    for name, preds in (("rule-based", rule_tier), ("semantic", sem_tier)):
        errors = tier_error_breakdown(expected_tier, preds)
        out.append(f"  {name}: over-routed (expected cheap -> strong) = {errors['over_routed']}, "
                   f"under-routed (expected strong -> cheap) = {errors['under_routed']}")
    differing = sum(a != b for a, b in zip(rule_tier, sem_tier))
    out += [f"  Prompts where the two routers chose different tiers: {differing}",
            "  (Both routers use the SAME complexity rules for the tier, so this should be 0.)",
            "", "Tier confusion - rule-based (rows = expected, columns = predicted)",
            format_confusion_matrix(confusion_matrix(expected_tier, rule_tier, TIERS), TIERS), ""]

    # --- disagreements
    breakdown = disagreement_breakdown(results)
    out += ["4. ROUTER DISAGREEMENT (category)", "-" * 78,
            f"  Different category chosen: {breakdown['total']} of {n} prompts",
            f"    only rule-based matched the label: {breakdown['rule_only_correct']}",
            f"    only semantic matched the label:   {breakdown['semantic_only_correct']}",
            f"    both wrong:                        {breakdown['both_wrong']}"]
    lines = [f"  [{r.record['id']}] expected={r.record['expected_category']:<16} rule={r.rule.category:<16} "
             f"semantic={r.semantic.category:<16} | {_short(r.record['prompt'], 40)}"
             for r in category_disagreements(results)]
    out += _list_items(lines, max_listed) + [""]

    # --- similarity
    fallbacks = [r for r in results if r.semantic.similarity is not None and r.semantic.similarity < MIN_SIMILARITY]
    low = low_similarity_results(results, low_threshold)
    mean_right, mean_wrong = mean_similarity_by_correctness(results)
    out += ["5. SEMANTIC SIMILARITY", "-" * 78,
            f"  Below the router's own MIN_SIMILARITY ({MIN_SIMILARITY}) -> fell back to 'general': {len(fallbacks)}",
            f"  Below the reporting threshold {low_threshold} (lowest first): {len(low)}"]
    lines = [f"  [{r.record['id']}] sim={r.semantic.similarity:.3f} expected={r.record['expected_category']:<16} "
             f"semantic={r.semantic.category:<16} {'ok' if r.semantic_category_correct else 'WRONG'}"
             for r in low]
    out += _list_items(lines, max_listed)
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    out += [f"  Mean similarity when semantic category was right: {fmt(mean_right)}; when wrong: {fmt(mean_wrong)}",
            "  (A useful score should be clearly lower when the router is wrong.)", ""]

    # --- misclassified
    out += ["6. PROMPTS TO INSPECT MANUALLY", "-" * 78, "Category errors - rule-based:"]
    wrong = [r for r in results if not r.rule_category_correct]
    lines = [f"  [{r.record['id']}] expected={r.record['expected_category']} predicted={r.rule.category} | {_short(r.record['prompt'])}"
             for r in wrong]
    out += _list_items(lines or ["  none"], max_listed)
    out += ["", "Category errors - semantic:"]
    wrong = [r for r in results if not r.semantic_category_correct]
    lines = [f"  [{r.record['id']}] expected={r.record['expected_category']} predicted={r.semantic.category} "
             f"(sim {r.semantic.similarity:.3f}) | {_short(r.record['prompt'])}" for r in wrong]
    out += _list_items(lines or ["  none"], max_listed)
    out += ["", "Tier errors - rule-based:"]
    wrong = [r for r in results if not r.rule_tier_correct]
    lines = [f"  [{r.record['id']}] expected={r.record['expected_tier']} predicted={r.rule.tier} | {_short(r.record['prompt'])}"
             for r in wrong]
    out += _list_items(lines or ["  none"], max_listed)
    if rule_tier == sem_tier:
        out += ["", "Tier errors - semantic: identical to rule-based (same tier policy)."]
    else:
        out += ["", "Tier errors - semantic:"]
        wrong = [r for r in results if not r.semantic_tier_correct]
        lines = [f"  [{r.record['id']}] expected={r.record['expected_tier']} predicted={r.semantic.tier} | {_short(r.record['prompt'])}"
                 for r in wrong]
        out += _list_items(lines or ["  none"], max_listed)

    out += ["", bar, "HOW TO READ THIS",
            "  - n is small, so differences of a few prompts are within noise (see the 95% CIs).",
            "  - Labels are human judgements; review the 'inspect manually' lists before trusting any number.",
            "  - Do not tune the router on these same prompts and then quote the score as performance.",
            "  - This measures routing decisions only, not answer quality, latency or cost.", bar]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "id", "expected_category", "expected_tier", "tags", "prompt", "label_note",
    "rule_category", "rule_tier", "rule_category_correct", "rule_tier_correct", "rule_reason",
    "semantic_category", "semantic_tier", "semantic_similarity",
    "semantic_category_correct", "semantic_tier_correct", "semantic_reason",
    "categories_agree", "low_similarity",
]


def _flag(value: bool) -> str:
    return "true" if value else "false"


def write_csv(results: Sequence[PromptResult], path: Path | str, low_threshold: float = DEFAULT_LOW_SIMILARITY) -> Path:
    """Write one row per prompt. Creates the parent folder. UTF-8 with BOM so Excel opens it cleanly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            sim = r.semantic.similarity
            writer.writerow({
                "id": r.record["id"],
                "expected_category": r.record["expected_category"],
                "expected_tier": r.record["expected_tier"],
                "tags": "|".join(r.record.get("tags", [])),
                "prompt": r.record["prompt"],
                "label_note": r.record["label_note"],
                "rule_category": r.rule.category,
                "rule_tier": r.rule.tier,
                "rule_category_correct": _flag(r.rule_category_correct),
                "rule_tier_correct": _flag(r.rule_tier_correct),
                "rule_reason": r.rule.reason,
                "semantic_category": r.semantic.category,
                "semantic_tier": r.semantic.tier,
                "semantic_similarity": "" if sim is None else f"{sim:.4f}",
                "semantic_category_correct": _flag(r.semantic_category_correct),
                "semantic_tier_correct": _flag(r.semantic_tier_correct),
                "semantic_reason": r.semantic.reason,
                "categories_agree": _flag(r.categories_agree),
                "low_similarity": _flag(sim is not None and sim < low_threshold),
            })
    return path


# ---------------------------------------------------------------------------
# 6. Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.benchmark_routing",
        description="Compare the rule-based router with the semantic-plus-rules router on a labelled dataset "
                    "(no LLM or paid API calls).",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="input JSONL file (default: data/routing_benchmark.jsonl)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output CSV file (default: reports/routing_benchmark_results.csv)")
    parser.add_argument("--low-similarity-threshold", type=float, default=DEFAULT_LOW_SIMILARITY,
                        help=f"report semantic scores below this value (default: {DEFAULT_LOW_SIMILARITY})")
    parser.add_argument("--max-listed", type=int, default=40,
                        help="maximum prompts printed per list (the CSV always has every prompt)")
    args = parser.parse_args(argv)
    if not 0 < args.low_similarity_threshold <= 1:
        parser.error("--low-similarity-threshold must be greater than 0 and at most 1")
    if args.max_listed < 1:
        parser.error("--max-listed must be at least 1")
    return args


def main(
    argv: Sequence[str] | None = None,
    rule_router: Router | None = None,
    semantic_router: Router | None = None,
) -> int:
    """Run the benchmark. The router arguments exist so tests can use fakes."""
    try:  # avoid crashes on consoles that cannot print some characters
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    args = parse_args(argv)

    try:
        records = load_dataset(args.dataset)
        near_duplicates = check_leakage(records)
    except DatasetError as exc:
        print(f"ERROR: invalid dataset ({args.dataset})\n{exc}", file=sys.stderr)
        return 2

    print(f"Loaded {len(records)} prompts from {args.dataset}")
    print("Running both routers. No LLM or paid API is called.")
    if semantic_router is None:
        print("Loading the local embedding model (the first run downloads about 90 MB once)...")
    try:
        results = run_benchmark(records, rule_router, semantic_router)
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    report = render_report(
        results,
        dataset_label=str(args.dataset),
        dataset_sha256=dataset_fingerprint(args.dataset),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        low_threshold=args.low_similarity_threshold,
        max_listed=args.max_listed,
        near_duplicates=near_duplicates,
    )
    print()
    print(report)

    try:
        written = write_csv(results, args.output, args.low_similarity_threshold)
    except OSError as exc:
        print(f"ERROR: could not write the CSV to {args.output}: {type(exc).__name__}", file=sys.stderr)
        return 4
    print(f"\nPer-prompt CSV written to: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
