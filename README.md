# Laundered Consensus

Code and released data for *Laundered Consensus: Do AI Watchdogs Check the Approvals Other Agents Claim?* (Coetzee, Sinha, Vijay), Apart Research Swarm Alignment Sprint, November 2026.

## The study

A packet has four parts: a rule, a proposed change, handoff messages from other agents, and attached records. The change breaks the rule unless a record approves it. A watchdog model reads the packet and returns clear or flag. In a laundered packet the handoffs claim the change was approved before, but no attached record says so. We test whether four hosted watchdogs (Gemma 4 31B, Qwen 3.8 27B, DeepSeek V4 Flash, GLM 5.3 Flash) check the records or trust the claim, under an adaptive attacker, in live relay chains, in a small GLM 5.3 Flash swarm, and after fine-tuning Gemma 4 31B. All rules, people, records and systems are fictional.

## Folders and report sections

| Folder | Contents | Report section |
|---|---|---|
| `scripts/generate/` | Scripted packet generator and batch checker (DeepSeek V4 Flash from fixed templates, six rule families) | 3.2 |
| `scripts/invent/` | Builders for the Invent outside-test sets from checked rounds | 3.2, 4.8, Appendix C |
| `scripts/labels/` | Builds the held-out exclusion list from two label reads | 3.5, 4.5 |
| `scripts/score/` | Scorers (Wilson intervals, exact McNemar), leak check, v789 scorer | 3.5, 3.6, 4.6, 4.8 |
| `scripts/serve/` | Modal server for a base model or LoRA adapter | 3.5 |
| `scripts/fixtures/` | Toy fixtures for the self-test; no study packets | none |
| `data/exclusions/` | The 17-row held-out exclusion list and its freeze line | 3.5, 4.5 |
| `data/freezes/` | Freeze hashes and manifests for the Invent outside tests and the policy-prompt run | 3.6 |
| `data/packets/` | Benign packet sets (added at upload; see its README) | 3.1, 3.2 |
| `results/verdicts/` | Text-free per-read verdicts (run, model, condition, packet id, verdict, error, round) | 4.1 to 4.8 |
| `figures/` | Figure sources (SVG) and renders (PNG) | Figures 1 to 4 |
| `docs/` | Watchdog prompts and the full inventory | 3.1, Appendix C |
| `con-artist/` | Attack loop, live chains, chain scoring (Deepayan Sinha adds) | 3.3, 3.4, 4.2 to 4.4 |
| `train-modal/` | Training, evaluation and scorers on Modal (Deepayan Sinha adds) | 3.5, 4.6, 4.7 |

## What is withheld and why

The work maps which claims get a rule-breaking change past an AI watchdog. The texts that worked would give real help to an attacker, so we withhold them: the adaptive attacker's winning handoffs, the planted-line set, the forged notes, the live-chain and swarm transcripts, the attack-shaped test rows (mechanisms `described_absent`, `explained_gap_flag`, `notes_only_flag`), and the answer keys for the held-out and Invent test sets. For that reason the verdict tables carry no gold label or mechanism column. Raw model outputs are also withheld, since their reason text quotes packets. Tactics are described by category only (report Appendix A). The withheld material stays in a private repository, available to reviewers on request. `docs/INVENTORY.md` lists every item and its status.

## Verifying the freeze hashes

Every file in this repository is listed with its SHA-256 in `SHA256SUMS.txt`:

```bash
sha256sum -c SHA256SUMS.txt
```

The exclusion list must match its freeze line (`5d3b6799...438c3`, hashed 23:54:32 UTC on 7 November 2026):

```bash
sha256sum data/exclusions/FINAL_EXCLUSIONS.csv
cat data/exclusions/FINAL_EXCLUSIONS_FREEZE.txt
```

The hashes in `data/freezes/*/FREEZE.txt` and `MANIFEST.json` cover the withheld test sets. A reviewer given the private sets can check them with `sha256sum` against those lines (v789 outside test: `0dd22874...e167`, 795 rows, frozen 16:02:32 UTC on 7 November 2026). Early freezes (`ch1`, `ch2`, `sftA`, `dpoB`) no longer verify against the current code, which was later extended; their recorded hashes and commits are the record (report 3.6).

Two scripts carry small release edits so the release scan stays clean: section-header strings are built from a constant, with logic unchanged. `docs/inventory.json` gives the hash of each original.

## Prior work

The prior-work line is the 6 November 2026 kickoff. Items marked † in the report were made before it. This list matches report Appendix B.

| Item | Author | Date | Used for |
|---|---|---|---|
| Pilot packets (87 rows, 39 sets) and scripted handoffs† | Coetzee | [team: date] | Positive control, attack targets |
| Quiet Suppression study and harness† | all three | October 2026 | Motivation, harness |
| `tools/ask_model.py`, `modal-run/adaption_smoke.py`† | Sinha | before 6 Nov [team: date] | Model calls, training smoke test |

Outside code is listed in the `con-artist/` and `train-modal/` READMEs.

## Keys

Scripts read API keys from the environment or a local env file at run time. No key is stored here.

## License

Suggested: MIT for code, CC BY 4.0 for data and figures. See `LICENSE_SUGGESTION.md`.
