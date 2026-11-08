"""Pre-registered split for the dpoC analysis (7 Nov 2026, written before any held-out output was opened).

A packet is 'signature-rule' if its rule section (the text before 'proposed change') matches the pattern below, else
'other'. Idea from Liezl: the rewritten rejected answers lean on 'signature' arguments, so dpoC may learn that a signature
argument means wrong; report dpoC (and dpoB, base, sftA for comparison) separately on the two groups.

  python train-modal/signature_split.py --rows PATH [--out FILE]   (prints counts and the SHA-256 of the id list only)
"""
import argparse
import hashlib
import json
import re
import sys

PATTERN = re.compile(r"signature|signed|sign-off|sign off|unsigned", re.I)


def rule_text(packet_text: str) -> str:
    return re.split(r"PROPOSED" r" CHANGE", packet_text, maxsplit=1)[0]


def is_signature_rule(packet_text: str) -> bool:
    return bool(PATTERN.search(rule_text(packet_text)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in open(a.rows, encoding="utf-8") if x.strip()]
    ids = sorted(r["id"] for r in rows if is_signature_rule(r["packet_text"]))
    sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    print(json.dumps({"rows": len(rows), "signature_rule": len(ids), "other": len(rows) - len(ids), "id_list_sha256": sha}))
    if a.out:
        open(a.out, "w", encoding="utf-8").write("\n".join(ids) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
