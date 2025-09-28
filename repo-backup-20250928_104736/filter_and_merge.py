#!/usr/bin/env python3
# filter_and_merge.py  safe filter+merge for pilots
import json, sys
from pathlib import Path

synthetic = Path("datasets/synthetic/cleaned/22sectors.jsonl")
pilots = {
    "finans": Path("datasets/pilots/cleaned_finans.jsonl"),
    "saglik": Path("datasets/pilots/cleaned_saglik.jsonl"),
    "egitim": Path("datasets/pilots/cleaned_egitim.jsonl"),
}


def iter_jsonl(p):
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"Warning: skip line {i} in {p}: {e}", file=sys.stderr)


if not synthetic.exists():
    print("ERROR: synthetic missing:", synthetic, file=sys.stderr)
    sys.exit(1)

syn = list(iter_jsonl(synthetic))
print(f"Loaded {len(syn)} synthetic records from {synthetic}")

for sname, pfile in pilots.items():
    out = pfile.parent / f"pilot_{sname}.jsonl"
    kept = 0
    written = 0
    with out.open("w", encoding="utf-8") as of:
        if pfile.exists():
            for obj in iter_jsonl(pfile):
                of.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
        for obj in syn:
            if sname in str(obj.get("sector", "")).lower():
                of.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
                written += 1
    print(
        f"Wrote {out} (pilot_lines={written-kept}, synthetic_appended={kept}, total={written})"
    )
