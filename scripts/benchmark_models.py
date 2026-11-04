
"""
Runs a blinded paired-model evaluation on the 40-prompt RouteWise dataset.

Each prompt is sent to both configured models:
- cheap: Gemini Flash-Lite
- strong: Gemini Flash

Produces:
- A blinded CSV containing prompts and responses.
- A key CSV containing model identities and performance metadata.
"""

import csv
import json
import random
import time
from pathlib import Path

from app.config import MODEL_BY_TIER
from app.dispatcher import DispatchError, dispatch_to_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "evaluation_dataset.jsonl"
REPORT_DIR = PROJECT_ROOT / "reports"

EXPECTED_RECORDS = 40
EXPECTED_TIERS = {"cheap", "strong"}
INFERENCE_CONFIG = "native_defaults"
REQUEST_DELAY_SECONDS = 2


# --------------------------------------------------
# Dataset validation
# --------------------------------------------------

def load_and_validate_dataset() -> list[dict]:
    """Load the evaluation dataset and validate its structure."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATA_FILE}"
        )

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        tasks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(tasks) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records, found {len(tasks)}."
        )

    required_fields = {
        "id",
        "category",
        "prompt",
        "source_dataset",
        "source_id",
        "evaluation_method",
    }

    unique_ids = set()
    gsm8k_count = 0
    ifeval_count = 0

    for task in tasks:

        missing_fields = required_fields - task.keys()

        if missing_fields:
            raise ValueError(
                f"Task is missing fields: {missing_fields}"
            )

        task_id = task["id"]

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Found a missing or invalid task ID.")

        if task_id in unique_ids:
            raise ValueError(f"Duplicate task ID: {task_id}")

        unique_ids.add(task_id)

        if not isinstance(task["prompt"], str) or not task["prompt"].strip():
            raise ValueError(f"Empty prompt: {task_id}")

        source = task["source_dataset"]

        if source == "openai/gsm8k":
            gsm8k_count += 1

            if not task.get("reference_answer"):
                raise ValueError(
                    f"GSM8K reference answer missing: {task_id}"
                )

            if task["evaluation_method"] != "numeric_match":
                raise ValueError(
                    f"Unexpected GSM8K evaluation method: {task_id}"
                )

        elif source == "google/IFEval":
            ifeval_count += 1

            instruction_ids = task.get("instruction_id_list")
            kwargs = task.get("kwargs")

            if not isinstance(instruction_ids, list) or not instruction_ids:
                raise ValueError(
                    f"IFEval instruction metadata missing: {task_id}"
                )

            if not isinstance(kwargs, list):
                raise ValueError(
                    f"IFEval kwargs missing: {task_id}"
                )

            if len(instruction_ids) != len(kwargs):
                raise ValueError(
                    f"IFEval metadata length mismatch: {task_id}"
                )

            if task["evaluation_method"] != "ifeval_strict":
                raise ValueError(
                    f"Unexpected IFEval evaluation method: {task_id}"
                )

        else:
            raise ValueError(
                f"Unknown dataset source: {source}"
            )

    if gsm8k_count != 20:
        raise ValueError(
            f"Expected 20 GSM8K records, found {gsm8k_count}."
        )

    if ifeval_count != 20:
        raise ValueError(
            f"Expected 20 IFEval records, found {ifeval_count}."
        )

    print(
        "Dataset validation passed: "
        "40 unique prompts (20 GSM8K, 20 IFEval)."
    )

    return tasks


# --------------------------------------------------
# Model configuration validation
# --------------------------------------------------

def validate_model_configuration() -> None:
    """Ensure exactly two distinct configured models are available."""

    if set(MODEL_BY_TIER.keys()) != EXPECTED_TIERS:
        raise ValueError(
            f"Expected exactly these model tiers: {EXPECTED_TIERS}. "
            f"Found: {set(MODEL_BY_TIER.keys())}"
        )

    cheap_model = MODEL_BY_TIER["cheap"]
    strong_model = MODEL_BY_TIER["strong"]

    if not isinstance(cheap_model, str) or not cheap_model.strip():
        raise ValueError("Invalid cheap model configuration.")

    if not isinstance(strong_model, str) or not strong_model.strip():
        raise ValueError("Invalid strong model configuration.")

    if cheap_model == strong_model:
        raise ValueError("Both tiers point to the same model.")

    print("Model configuration validated.")

    print(f"Cheap model : {cheap_model}")
    print(f"Strong model: {strong_model}")


# --------------------------------------------------
# Benchmark execution
# --------------------------------------------------

def run_evaluation() -> None:

    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    # Unique run identifier prevents accidental overwriting.
    run_id = time.time_ns()

    blinded_file = (
        REPORT_DIR / f"model_benchmark_40_blind_{run_id}.csv"
    )

    key_file = (
        REPORT_DIR / f"model_benchmark_40_key_{run_id}.csv"
    )

    # Validate everything before making API calls.
    tasks = load_and_validate_dataset()
    validate_model_configuration()

    # Record the seed so the randomisation can be reproduced.
    seed = random.SystemRandom().randrange(0, 2**32)
    random.seed(seed)

    print(f"\nExperiment initialized. Random seed: {seed}")
    print(f"Total prompts: {len(tasks)}")
    print(f"Expected model calls: {len(tasks) * 2}")

    # --------------------------------------------------
    # CSV structures
    # --------------------------------------------------

    blinded_fieldnames = [
        "task_id",
        "category",
        "source_dataset",
        "prompt",
        "Model_A_answer",
        "Model_B_answer",
    ]

    key_fieldnames = [
        "task_id",
        "source_dataset",
        "source_id",
        "evaluation_method",
        "random_seed",
        "inference_config",
        "execution_order",

        "Model_A_tier",
        "Model_A_id",
        "Model_A_latency_ms",
        "Model_A_tokens",
        "Model_A_error",

        "Model_B_tier",
        "Model_B_id",
        "Model_B_latency_ms",
        "Model_B_tokens",
        "Model_B_error",
    ]

    # --------------------------------------------------
    # Telemetry counters
    # --------------------------------------------------

    total_calls = 0
    successes = 0
    failures = 0

    expected_calls = len(tasks) * 2

    # --------------------------------------------------
    # Execute paired benchmark
    # --------------------------------------------------

    with (
        open(blinded_file, "w", newline="", encoding="utf-8") as fb,
        open(key_file, "w", newline="", encoding="utf-8") as fk,
    ):

        writer_b = csv.DictWriter(
            fb,
            fieldnames=blinded_fieldnames
        )

        writer_k = csv.DictWriter(
            fk,
            fieldnames=key_fieldnames
        )

        writer_b.writeheader()
        writer_k.writeheader()

        fb.flush()
        fk.flush()

        for index, task in enumerate(tasks, start=1):

            print(
                f"\nPrompt {index}/{len(tasks)} "
                f"| {task['id']} "
                f"| {task['source_dataset']}"
            )

            # Randomly assign model tiers to labels A and B.
            shuffled_tiers = ["cheap", "strong"]
            random.shuffle(shuffled_tiers)

            label_to_tier = {
                "A": shuffled_tiers[0],
                "B": shuffled_tiers[1],
            }

            # Randomise which model is called first.
            execution_order = ["A", "B"]
            random.shuffle(execution_order)

            execution_order_str = (
                f"{execution_order[0]}_then_{execution_order[1]}"
            )

            results = {}

            for label in execution_order:

                tier = label_to_tier[label]
                model_id = MODEL_BY_TIER[tier]

                print(
                    f"  -> Model {label}: {model_id} ... ",
                    end="",
                    flush=True
                )

                total_calls += 1

                start_time = time.perf_counter()

                try:
                    # Send ONLY the original prompt text.
                    # Reference answers and evaluation metadata
                    # are never passed to the model.
                    response = dispatch_to_model(
                        model_id,
                        task["prompt"]
                    )

                    answer = response.answer

                    usage = response.usage

                    tokens = (
                        usage.total_tokens
                        if usage is not None
                        else None
                    )

                    error = None
                    successes += 1

                    print("SUCCESS")

                except DispatchError as exc:

                    answer = f"ERROR: {exc.category}"
                    tokens = None
                    error = exc.category

                    failures += 1

                    print(f"FAILED ({exc.category})")

                latency_ms = round(
                    (time.perf_counter() - start_time) * 1000
                )

                results[label] = {
                    "tier": tier,
                    "id": model_id,
                    "answer": answer,
                    "latency_ms": latency_ms,
                    "tokens": tokens,
                    "error": error,
                }

                # Reduce request bursts.
                # No unnecessary sleep after the final call.
                if total_calls < expected_calls:
                    time.sleep(REQUEST_DELAY_SECONDS)

            # --------------------------------------------------
            # Save completed prompt results immediately.
            # --------------------------------------------------

            writer_b.writerow({
                "task_id": task["id"],
                "category": task["category"],
                "source_dataset": task["source_dataset"],
                "prompt": task["prompt"],
                "Model_A_answer": results["A"]["answer"],
                "Model_B_answer": results["B"]["answer"],
            })

            fb.flush()

            writer_k.writerow({
                "task_id": task["id"],
                "source_dataset": task["source_dataset"],
                "source_id": task["source_id"],
                "evaluation_method": task["evaluation_method"],
                "random_seed": seed,
                "inference_config": INFERENCE_CONFIG,
                "execution_order": execution_order_str,

                "Model_A_tier": results["A"]["tier"],
                "Model_A_id": results["A"]["id"],
                "Model_A_latency_ms": results["A"]["latency_ms"],
                "Model_A_tokens": results["A"]["tokens"],
                "Model_A_error": results["A"]["error"],

                "Model_B_tier": results["B"]["tier"],
                "Model_B_id": results["B"]["id"],
                "Model_B_latency_ms": results["B"]["latency_ms"],
                "Model_B_tokens": results["B"]["tokens"],
                "Model_B_error": results["B"]["error"],
            })

            fk.flush()

            print(f"  Saved results for {task['id']}.")

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 45)
    print("ROUTEWISE MODEL EVALUATION COMPLETE")
    print("=" * 45)

    print(f"Total API calls : {total_calls}")
    print(f"Successful calls: {successes}")
    print(f"Failed calls    : {failures}")

    print(f"\nBlinded results: {blinded_file}")
    print(f"Answer key     : {key_file}")

    if total_calls != expected_calls:
        print(
            "\nWARNING: The number of completed calls "
            "differs from the expected number."
        )


if __name__ == "__main__":
    run_evaluation()
