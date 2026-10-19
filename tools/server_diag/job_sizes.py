#!/usr/bin/env python3
"""Run ON the server. Disk usage per job (read-only), biggest first.

  sudo python3 job_sizes.py [--jobs-dir /data/jobs] [--top 40]

Shows size, status, creation date and name, plus how much of each job is intermediate Brush exports
(model_<step>.ply every export_every steps, except the last one) - historically ~58% of the data disk.
Deleting whole jobs goes through the API (DELETE /jobs/{id}); deleting only intermediate exports needs a
shell on the server (the API cannot do it) and was never done - ask the owner first.
"""
import argparse
import json
import os
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--jobs-dir", default=os.environ.get("GS_JOBS_DIR", "/data/jobs"))
ap.add_argument("--top", type=int, default=40)
a = ap.parse_args()

out = subprocess.run(["du", "-s", "--block-size=1M", *[os.path.join(a.jobs_dir, d) for d in os.listdir(a.jobs_dir)]],
                     capture_output=True, text=True).stdout
rows = []
for line in out.splitlines():
    sz, path = line.split("\t", 1)
    name, status, created = "?", "?", "?"
    try:
        s = json.load(open(os.path.join(path, "state.json")))
        name, status, created = s.get("name", "?"), s.get("status", "?"), (s.get("created_at") or "?")[:10]
    except Exception:
        pass
    inter = 0
    od = os.path.join(path, "output")
    if os.path.isdir(od):
        raw = {int(m.group(1)): os.path.getsize(os.path.join(od, f)) / 1048576
               for f in os.listdir(od) for m in [re.fullmatch(r"model_(\d+)\.ply", f)] if m}
        if raw:
            inter = sum(v for k, v in raw.items() if k != max(raw))
    rows.append((int(sz), inter, os.path.basename(path), name, status, created))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print(f"TOTAL {tot / 1024:.0f} GB in {len(rows)} jobs; intermediate exports {sum(r[1] for r in rows) / 1024:.0f} GB")
print("size_GB inter_GB id8      status          created     name")
for sz, inter, jid, name, status, created in rows[:a.top]:
    print(f"{sz / 1024:7.1f} {inter / 1024:8.1f} {jid[:8]}  {status:<14}  {created}  {name}")
