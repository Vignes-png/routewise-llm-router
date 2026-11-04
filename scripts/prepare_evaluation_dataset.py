
"""
Prepare a reproducible 40-prompt evaluation dataset from GSM8K and IFEval.
"""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_FILE = DATA_DIR / "evaluation_dataset.jsonl"
METADATA_FILE = DATA_DIR / "evaluation_dataset_metadata.json"
README_FILE = DATA_DIR / "README_evaluation.md"

RANDOM_SEED = 42
SAMPLE_SIZE = 20

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
IFEVAL_REVISION = "b5fce8ae859b860d95b131137da74f013e1ec333"


def generate_gsm8k_id(question: str) -> str:
    """Generate a stable ID using the original question."""
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return f"gsm8k_{digest[:12]}"


def prepare_datasets():
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    records = []

    metadata = {
        "seed": RANDOM_SEED,
        "sample_size_per_dataset": SAMPLE_SIZE,
        "sources": {
            "gsm8k": {
                "dataset": "openai/gsm8k",
                "split": "test",
                "revision": GSM8K_REVISION,
                "license": "MIT",
                "selected_ids": []
            },
            "ifeval": {
                "dataset": "google/IFEval",
                "split": "train",
                "revision": IFEVAL_REVISION,
                "license": "Apache-2.0",
                "raw_rows": None,
                "unique_rows": None,
                "duplicates_removed": None,
                "selected_ids": []
            }
        }
    }

    # --------------------------------------------------
    # 1. GSM8K
    # --------------------------------------------------

    print("Loading GSM8K...")

    gsm8k = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision=GSM8K_REVISION
    )

    if len(gsm8k) < SAMPLE_SIZE:
        raise ValueError("GSM8K contains fewer than 20 examples.")

    gsm8k_sample = (
        gsm8k.shuffle(seed=RANDOM_SEED)
        .select(range(SAMPLE_SIZE))
    )

    for row in gsm8k_sample:
        question = row["question"].strip()
        answer = row["answer"]

        if "####" not in answer:
            raise ValueError(
                f"GSM8K answer extraction failed for question: {question[:100]}"
            )

        reference_answer = answer.split("####")[-1].strip()
        stable_id = generate_gsm8k_id(question)

        final_prompt = (
            question
            + "\n\nOn the final line, give your final numerical answer "
              "in this exact format: Final answer: <number>."
        )

        records.append({
            "id": stable_id,
            "category": "reasoning",
            "prompt": final_prompt,
            "source_dataset": "openai/gsm8k",
            "source_split": "test",
            "source_revision": GSM8K_REVISION,
            "source_id": stable_id,
            "evaluation_method": "numeric_match",
            "reference_answer": reference_answer
        })

        metadata["sources"]["gsm8k"]["selected_ids"].append(stable_id)

    # --------------------------------------------------
    # 2. IFEval: deduplicate BEFORE sampling
    # --------------------------------------------------

    print("Loading IFEval...")

    ifeval_raw = load_dataset(
        "google/IFEval",
        split="train",
        revision=IFEVAL_REVISION
    )

    first_row_by_key = {}
    unique_indices = []

    for index, row in enumerate(ifeval_raw):
        stable_id = str(row["key"])

        if stable_id not in first_row_by_key:
            first_row_by_key[stable_id] = row
            unique_indices.append(index)

        elif row != first_row_by_key[stable_id]:
            raise ValueError(
                f"Duplicate IFEval ID {stable_id} has conflicting contents."
            )

    ifeval = ifeval_raw.select(unique_indices)

    duplicate_count = len(ifeval_raw) - len(ifeval)

    print(f"Raw IFEval rows: {len(ifeval_raw)}")
    print(f"Unique IFEval prompts: {len(ifeval)}")
    print(f"Duplicate rows removed: {duplicate_count}")

    if len(ifeval) != 541:
        raise ValueError(
            f"Expected 541 unique IFEval prompts, found {len(ifeval)}."
        )

    metadata["sources"]["ifeval"]["raw_rows"] = len(ifeval_raw)
    metadata["sources"]["ifeval"]["unique_rows"] = len(ifeval)
    metadata["sources"]["ifeval"]["duplicates_removed"] = duplicate_count

    ifeval_sample = (
        ifeval.shuffle(seed=RANDOM_SEED)
        .select(range(SAMPLE_SIZE))
    )

    for row in ifeval_sample:
        stable_id = str(row["key"])
        prompt = row.get("prompt")
        instruction_ids = row.get("instruction_id_list")
        kwargs = row.get("kwargs")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"IFEval record {stable_id} has an empty prompt."
            )

        if not isinstance(instruction_ids, list) or not instruction_ids:
            raise ValueError(
                f"IFEval record {stable_id} has invalid instruction metadata."
            )

        if not isinstance(kwargs, list) or len(instruction_ids) != len(kwargs):
            raise ValueError(
                f"IFEval record {stable_id} has mismatched instruction metadata."
            )

        records.append({
            "id": stable_id,
            "category": "instruction_following",
            "prompt": prompt,
            "source_dataset": "google/IFEval",
            "source_split": "train",
            "source_revision": IFEVAL_REVISION,
            "source_id": stable_id,
            "evaluation_method": "ifeval_strict",
            "instruction_id_list": instruction_ids,
            "kwargs": kwargs
        })

        metadata["sources"]["ifeval"]["selected_ids"].append(stable_id)

    # --------------------------------------------------
    # 3. Validate the final dataset
    # --------------------------------------------------

    print("Validating dataset constraints...")

    if len(records) != 40:
        raise ValueError(
            f"Expected 40 records, generated {len(records)}."
        )

    ids = [record["id"] for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs detected in final dataset.")

    for record in records:
        if not record["prompt"].strip():
            raise ValueError(
                f"Empty prompt detected: {record['id']}"
            )

    gsm8k_count = sum(
        record["source_dataset"] == "openai/gsm8k"
        for record in records
    )

    ifeval_count = sum(
        record["source_dataset"] == "google/IFEval"
        for record in records
    )

    if gsm8k_count != 20 or ifeval_count != 20:
        raise ValueError(
            f"Incorrect dataset composition: "
            f"GSM8K={gsm8k_count}, IFEval={ifeval_count}"
        )

    # --------------------------------------------------
    # 4. Write files
    # --------------------------------------------------

    print(f"Writing dataset to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Writing metadata to {METADATA_FILE}...")

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    readme_content = """# RouteWise Evaluation Dataset

## Composition

### GSM8K: 20 prompts
- Source: openai/gsm8k
- Split: test
- Evaluation: numeric_match
- Reference answers are preserved.
- Prompts are adapted by appending a final-answer formatting instruction.

### IFEval: 20 prompts
- Source: google/IFEval
- Split: train
- Evaluation: ifeval_strict
- Original prompts and evaluation metadata are preserved.

At the pinned revision, IFEval loaded 1,082 rows containing
541 unique prompt IDs. Each prompt appeared twice with identical
contents. Duplicate rows were removed using the original IFEval
key before sampling.

## Evaluation methodology

This script only prepares the dataset. It does not evaluate
model responses.

- GSM8K requires numerical answer extraction and comparison.
- IFEval requires the separate official strict evaluator.
- Results must be reported separately because the datasets
  measure different capabilities.

## Reproducibility

Random seed: 42

Dataset revisions are pinned in this script and recorded
in evaluation_dataset_metadata.json.

The selected source IDs are also recorded in the metadata file.
"""

    print(f"Writing README to {README_FILE}...")

    with open(README_FILE, "w", encoding="utf-8") as file:
        file.write(readme_content)

    print("\nDataset preparation complete!")
    print(f"Total prompts: {len(records)}")
    print(f"GSM8K: {gsm8k_count}")
    print(f"IFEval: {ifeval_count}")


if __name__ == "__main__":
    prepare_datasets()
