# train-modal

Training, evaluation and scoring code on Modal for the Swarm Alignment Sprint (6 to 8 November 2026): report sections 3.5, 4.6 and 4.7. **All code here is sprint work written on or after 6 November 2026.** Fictional packets only.

## What it does

Gemma 4 31B (LoRA, one H200) is trained on a frozen training split in two ways: SFT (`sftA`), then DPO from the SFT adapter (`dpoB`, the main result). An exploratory second DPO run on rewritten rejected answers (`dpoC`) is reported apart. Base, SFT and DPO are then scored on a held-out set under the plain prompt, after the exclusion list was hashed, and on an outside test set from a different generator (`evalV789`).

## What is not in this folder

The training and held-out packet files, the answer keys, raw generations and run logs are not released: the packet files carry the packet section headers and the outside-test rows are attack-shaped. This folder holds code, settings, freezes, leak and prep summaries, and text-free result files (counts, Wilson intervals, training curves).

## Release edit

Packet section headers appear in some code and tests as string literals. For this release they are split inside the literals and written in lower case in comments and docstrings; every edited file gives the same syntax tree as the original. Original and released sha256 values are in `../docs/inventory-ours.json`, so the code hashes in `freezes/*.json` do not match this copy. With the private inputs in place, the 38 offline tests pass.

## Layout

| File | Purpose |
|---|---|
| `prep.py` | validate rows, build SFT and DPO records with the plain prompt, split by set; refuses held-out paths; needs a passing leak report |
| `leak_check.py` | laptop only: exact and word 8-gram checks of training rows against the held-out set, pilot and seed packets; fails closed |
| `freeze_train.py` | write and verify freeze files (hashes of code, config, prompt, data, leak report) |
| `score_eval.py` | Wilson intervals, missed violations, false alarms, parse failures, exact McNemar |
| `modal_train.py` | Modal app: `download`, `envcheck`, `onestep_*`, `sft`, `dpo`, `evaluate` |
| `make_eval_input.py`, `make_eval_input_v789.py` | build the evaluation inputs (the v789 builder refuses packets with metadata in the prompt) |
| `score_v789.py`, `score_v789_sensitivity.py` | score wrong verdicts per mechanism on the outside test |
| `score_signature.py`, `signature_split.py` | the pre-registered signature-rule split for `dpoC` (27 of 278 held-out rows) |
| `config/` | per-model settings |
| `freezes/` | freeze files for each run |
| `data/`, `data_dpoC/` | leak report and prep summary only |
| `results/` | text-free score and training-result files |
| `tests/` | offline tests |

## Keys

No key is stored here. Modal Secrets are referenced by name.
