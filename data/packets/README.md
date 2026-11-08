# Benign packet sets

The report releases the benign packet sets. They are not in this suggested tree, because every packet has the rule, change, handoff and record section headers, and this tree had to pass a zero-hit scan for those headers.

To add at upload, after a read:

- `train_v1_balanced.jsonl`: the 2,807-row scripted training split (from `sprint/train_bundle/`).
- `pilot_packets_87_with_labels.jsonl`: the 87-row pilot (prior work †).

Record each file's SHA-256 in `SHA256SUMS.txt` when it goes in.
