# Inventory

Every item from Liezl's side, marked include, redact or withhold. Source hashes are in `inventory.json`.

| Source | In repo | Action | Note |
|---|---|---|---|
| `sprint/gen/gen_triplets.py` | `scripts/generate/gen_triplets.py` | redact | section-header literals built from a constant; two prompt lines say "speaker-name prefix" instead of quoting the agent prefix; logic unchanged |
| `sprint/gen/check_batch.py` | `scripts/generate/check_batch.py` | redact | one header literal split; logic unchanged |
| `sprint/scripts/score.py` | `scripts/score/score.py` | include | scorer: Wilson intervals, McNemar |
| `sprint/scripts/check_leakage.py` | `scripts/score/check_leakage.py` | include | leak check between training and test sets |
| `sprint/runs/invent_v789_base/score789.py` | `scripts/score/score789.py` | include | v789 outside-test scorer; needs the withheld answer key to run |
| `sprint/scripts/modal_serve.py` | `scripts/serve/modal_serve.py` | include | serves a base model or adapter on Modal |
| `sprint/exclusions/build_final_exclusions.py` | `scripts/labels/build_final_exclusions.py` | include | joins the two label reads into the exclusion list |
| `sprint/invent_probe/assemble_v7.py` | `scripts/invent/assemble_v7.py` | include | builds an Invent outside-test set from checked rounds |
| `sprint/invent_probe/assemble_v789.py` | `scripts/invent/assemble_v789.py` | include | builds an Invent outside-test set from checked rounds |
| `sprint/invent_probe/assemble_combined.py` | `scripts/invent/assemble_combined.py` | include | builds an Invent outside-test set from checked rounds |
| `sprint/scripts/fixtures/self_test.py` | `scripts/fixtures/self_test.py` | include | toy self-test fixture, no study packets |
| `sprint/scripts/fixtures/prompt.txt` | `scripts/fixtures/prompt.txt` | include | toy self-test fixture, no study packets |
| `sprint/scripts/fixtures/prices.json` | `scripts/fixtures/prices.json` | include | toy self-test fixture, no study packets |
| `sprint/scripts/fixtures/responses.json` | `scripts/fixtures/responses.json` | include | toy self-test fixture, no study packets |
| `sprint/scripts/fixtures/packets.jsonl` | `scripts/fixtures/packets.jsonl` | include | toy self-test fixture, no study packets |
| `sprint/scripts/fixtures/heldout.jsonl` | `scripts/fixtures/heldout.jsonl` | include | toy self-test fixture, no study packets |
| `sprint/scripts/fixtures/train_clean.jsonl` | `scripts/fixtures/train_clean.jsonl` | include | toy self-test fixture, no study packets |
| `sprint/scripts/fixtures/train_leaky.jsonl` | `scripts/fixtures/train_leaky.jsonl` | include | toy self-test fixture, no study packets |
| `sprint/scripts/fixtures/conartist.jsonl` | `scripts/fixtures/conartist.jsonl` | include | toy self-test fixture, no study packets |
| `sprint/guard/prompts/prompt_plain.txt` | `docs/prompts/prompt_plain.txt` | include | watchdog prompt (PLAIN and the two policy prompts) |
| `sprint/guard/prompts/prompt_guard1.txt` | `docs/prompts/prompt_guard1.txt` | include | watchdog prompt (PLAIN and the two policy prompts) |
| `sprint/guard/prompts/prompt_guard3.txt` | `docs/prompts/prompt_guard3.txt` | include | watchdog prompt (PLAIN and the two policy prompts) |
| `sprint/exclusions/FINAL_EXCLUSIONS.csv` | `data/exclusions/FINAL_EXCLUSIONS.csv` | include | held-out exclusion list; hashes to the freeze line |
| `sprint/exclusions/FINAL_EXCLUSIONS_FREEZE.txt` | `data/exclusions/FINAL_EXCLUSIONS_FREEZE.txt` | include | freeze line for the list |
| `sprint/invent_probe/combined_v789/FREEZE.txt` | `data/freezes/invent_combined_v789/FREEZE.txt` | include | hash and counts only; the set itself is withheld |
| `sprint/invent_probe/combined_v789/MANIFEST.json` | `data/freezes/invent_combined_v789/MANIFEST.json` | include | hash and counts only; the set itself is withheld |
| `sprint/invent_probe/combined_v7/FREEZE.txt` | `data/freezes/invent_combined_v7/FREEZE.txt` | include | hash and counts only; the set itself is withheld |
| `sprint/invent_probe/combined_v7/MANIFEST.json` | `data/freezes/invent_combined_v7/MANIFEST.json` | include | hash and counts only; the set itself is withheld |
| `sprint/invent_probe/combined/FREEZE.txt` | `data/freezes/invent_combined/FREEZE.txt` | include | hash and counts only; the set itself is withheld |
| `sprint/invent_probe/combined/MANIFEST.json` | `data/freezes/invent_combined/MANIFEST.json` | include | hash and counts only; the set itself is withheld |
| `sprint/guard/FREEZE.txt` | `data/freezes/guard/FREEZE.txt` | redact | local absolute paths cut to repo-relative paths; hashes unchanged |
| `sprint/invent_probe_errata/ERRATA.md` | - | withhold | ties two attack-shaped row ids to gold and mechanism; report Appendix C carries the summary |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_errata_rescore.tsv` | `results/verdicts/verdicts_errata_rescore.tsv` | redact | 8 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_invent_test_base.tsv` | `results/verdicts/verdicts_invent_test_base.tsv` | redact | 624 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_invent_v789_base.tsv` | `results/verdicts/verdicts_invent_v789_base.tsv` | redact | 2556 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_invent_v789_random_reader.tsv` | `results/verdicts/verdicts_invent_v789_random_reader.tsv` | redact | 200 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_invent_v789_second_reader.tsv` | `results/verdicts/verdicts_invent_v789_second_reader.tsv` | redact | 256 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_invent_v7_base.tsv` | `results/verdicts/verdicts_invent_v7_base.tsv` | redact | 624 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_pilot_crafted.tsv` | `results/verdicts/verdicts_pilot_crafted.tsv` | redact | 200 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_pilot_labels.tsv` | `results/verdicts/verdicts_pilot_labels.tsv` | redact | 640 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_pilot_rerun.tsv` | `results/verdicts/verdicts_pilot_rerun.tsv` | redact | 38 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `sprint/kimi_review_bundle/out_reviewer/verdicts_pilot_scoring.tsv` | `results/verdicts/verdicts_pilot_scoring.tsv` | redact | 1044 rows; gold and mechanism columns dropped (answer key); reason text already absent |
| `report_sections/visuals/fig1_packet_and_triplet.svg` | `figures/fig1_packet_and_triplet.svg` | include | figure source |
| `report_sections/visuals/fig2_data_pipeline.svg` | `figures/fig2_data_pipeline.svg` | include | figure source |
| `report_sections/visuals/fig3_attack_setups.svg` | `figures/fig3_attack_setups.svg` | include | figure source |
| `report_sections/visuals/fig4_claim_vs_visibility.svg` | `figures/fig4_claim_vs_visibility.svg` | include | figure source |
| `report_sections/visuals/png/fig1_packet_and_triplet.png` | `figures/png/fig1_packet_and_triplet.png` | include | rendered figure |
| `report_sections/visuals/png/fig2_data_pipeline.png` | `figures/png/fig2_data_pipeline.png` | include | rendered figure |
| `report_sections/visuals/png/fig3_attack_setups.png` | `figures/png/fig3_attack_setups.png` | include | rendered figure |
| `report_sections/visuals/png/fig4_claim_vs_visibility.png` | `figures/png/fig4_claim_vs_visibility.png` | include | rendered figure |
| `sprint/train_bundle/train_v1_balanced.jsonl (2,807-row training split) and train_v1*.jsonl` | - | withhold | benign set planned for release; packet text carries the section markers, so Liezl adds it at upload after her own read |
| `sprint/pilot/bundle_for_deepayan/pilot_packets_87_with_labels.jsonl` | - | withhold | pilot packets (prior work); add at upload as above |
| `sprint/scripts/run_watchdogs.py, sprint/pilot/bundle_for_deepayan/run_watchdogs.py` | - | withhold | watchdog runner; one header literal to split and prompts to check before release |
| `sprint/gen/out/heldout_v*_DO_NOT_TRAIN.jsonl, sprint/guard/heldout_packets.jsonl` | - | withhold | held-out set with its answer key |
| `sprint/invent_probe/combined*/*.jsonl, sprint/invent_data/*, sprint/runs/*/packets*.jsonl` | - | withhold | Invent outside-test sets with answer keys; includes attack-shaped rows (described_absent, explained_gap_flag, notes_only_flag) |
| `sprint/invent_probe_errata/*_fixed.jsonl` | - | withhold | fixed attack-shaped test rows |
| `sprint/invent_probe/*prompt*, build_v*_prompts.py` | - | withhold | Invent generation prompts for attack-shaped mechanisms |
| `sprint/pilot/crafted/*` | - | withhold | hand-crafted attack handoffs and their logs |
| `sprint/chain_planted/*, planted_lines_v1_for_deepayan.zip` | - | withhold | planted lines |
| `GLM swarm folder (swarm code, threads, decisions, logs)` | - | withhold | swarm transcripts and forged notes; swarm code held until checked for embedded forged-note text |
| `sprint/runs/*/out/*, sprint/guard/out/*, sprint/**/log_*.txt` | - | withhold | raw model outputs whose reason text quotes packets |
| `sprint/heldout_audit/* (blind key, reader marks, label-check answers), sprint/exclusions/*marks*` | - | withhold | answer keys and reader marks that reveal labels |
| `sprint/kimi_review_bundle/*` | - | withhold | internal reviewer bundles |
| `C:/Users/Liezl/tools/.secrets and any env file` | - | withhold | API keys |
