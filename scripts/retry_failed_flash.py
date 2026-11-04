
"""Retry only failed Flash calls from the previous 40-prompt benchmark."""

import csv
import os
import random
import time
from pathlib import Path

from app.config import MODEL_BY_TIER
from app.dispatcher import DispatchError, dispatch_to_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

RUN_ID = "1793561062886796000"

ORIGINAL_BLIND = REPORT_DIR / f"model_benchmark_40_blind_{RUN_ID}.csv"
ORIGINAL_KEY = REPORT_DIR / f"model_benchmark_40_key_{RUN_ID}.csv"

RECOVERED_BLIND = REPORT_DIR / f"model_benchmark_40_blind_{RUN_ID}_recovered.csv"
RECOVERED_KEY = REPORT_DIR / f"model_benchmark_40_key_{RUN_ID}_recovered.csv"

DELAY_SECONDS = 20


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_csv(path, rows):
    """Atomically save results to avoid losing completed retries."""
    if not rows:
        return

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, path)


def main():
    if not ORIGINAL_BLIND.exists() or not ORIGINAL_KEY.exists():
        raise FileNotFoundError("Original benchmark CSV files are missing.")

    # Resume from recovered files if this script has run before.
    recovered_exists = (
        RECOVERED_BLIND.exists(),
        RECOVERED_KEY.exists()
    )

    if recovered_exists[0] != recovered_exists[1]:
        raise RuntimeError(
            "Only one recovered CSV exists. Inspect reports before continuing."
        )

    if all(recovered_exists):
        blind_rows = read_csv(RECOVERED_BLIND)
        key_rows = read_csv(RECOVERED_KEY)
        print("Resuming from previously recovered results.")
    else:
        blind_rows = read_csv(ORIGINAL_BLIND)
        key_rows = read_csv(ORIGINAL_KEY)

    if len(blind_rows) != 40 or len(key_rows) != 40:
        raise ValueError("Expected exactly 40 rows in both CSV files.")

    blind_by_id = {row["task_id"]: row for row in blind_rows}
    key_by_id = {row["task_id"]: row for row in key_rows}

    if len(blind_by_id) != 40 or len(key_by_id) != 40:
        raise ValueError("Duplicate task IDs found.")

    if set(blind_by_id) != set(key_by_id):
        raise ValueError("Blind CSV and key CSV task IDs do not match.")

    if "strong" not in MODEL_BY_TIER:
        raise ValueError("Strong model is missing from MODEL_BY_TIER.")

    strong_model = MODEL_BY_TIER["strong"]

    # Find only failed Flash requests.
    pending = []

    for row in key_rows:
        task_id = row["task_id"]

        for label in ("A", "B"):
            if row[f"Model_{label}_id"] != strong_model:
                continue

            error = row[f"Model_{label}_error"].strip()

            if error == "rate_limit":
                pending.append((task_id, label))

    print(f"Strong model: {strong_model}")
    print(f"Missing Flash responses: {len(pending)}")

    if not pending:
        print("No failed Flash calls remain.")
        return

    # Preserve the originals and create separate recovered copies.
    save_csv(RECOVERED_BLIND, blind_rows)
    save_csv(RECOVERED_KEY, key_rows)

    completed = 0
    stopped_by_rate_limit = False

    # Randomise retry order to avoid processing only one dataset first.
    seed = int(key_rows[0]["random_seed"])
    random.Random(seed + 1).shuffle(pending)

    for index, (task_id, label) in enumerate(pending, start=1):
        blind = blind_by_id[task_id]
        key = key_by_id[task_id]

        prompt = blind["prompt"]

        print(
            f"\nRetry {index}/{len(pending)} | "
            f"{task_id} | Model {label}"
        )

        start = time.perf_counter()

        try:
            response = dispatch_to_model(strong_model, prompt)

            answer = response.answer
            usage = response.usage

            tokens = (
                usage.total_tokens
                if usage is not None
                else None
            )

            error = None
            print("SUCCESS")

        except DispatchError as exc:
            answer = f"ERROR: {exc.category}"
            tokens = None
            error = exc.category

            print(f"FAILED: {error}")

        latency_ms = round((time.perf_counter() - start) * 1000)

        # Update only the previously failed model response.
        blind[f"Model_{label}_answer"] = answer

        key[f"Model_{label}_latency_ms"] = latency_ms
        key[f"Model_{label}_tokens"] = tokens if tokens is not None else ""
        key[f"Model_{label}_error"] = error or ""

        # Save both files after every attempt.
        save_csv(RECOVERED_BLIND, blind_rows)
        save_csv(RECOVERED_KEY, key_rows)

        if error is None:
            completed += 1

        # Stop immediately if the provider rate-limits us again.
        if error == "rate_limit":
            stopped_by_rate_limit = True
            print(
                "\nStopping because Flash returned another rate-limit error."
            )
            break

        if index < len(pending):
            print(f"Waiting {DELAY_SECONDS} seconds...")
            time.sleep(DELAY_SECONDS)

    print("\n" + "=" * 40)
    print("RECOVERY SUMMARY")
    print("=" * 40)
    print(f"Successful retries: {completed}")
    print(f"Remaining rate-limit failures: {sum(1 for r in key_rows for label in ('A', 'B') if r[f'Model_{label}_id'] == strong_model and r[f'Model_{label}_error'] == 'rate_limit')}")

    print(f"\nRecovered blind CSV: {RECOVERED_BLIND}")
    print(f"Recovered key CSV: {RECOVERED_KEY}")

    if stopped_by_rate_limit:
        print("\nRun this script again later to resume unfinished retries.")


if __name__ == "__main__":
    main()
