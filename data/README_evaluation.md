# RouteWise Evaluation Dataset

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
