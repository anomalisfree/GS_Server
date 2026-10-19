#!/usr/bin/env python3
"""Run ON the server. Live view of how the COLMAP mapper is doing, from the gs-server journal.

  sudo python3 colmap_models.py "2026-10-13 15:44:00"     # journal since that server (UTC) time
  sudo python3 colmap_models.py "-60 min"

Prints the number of registered images in every sub-model the mapper has started so far
([239] = one model with 239 images; [13, 4, 23, 11, 30] = many tiny fragments = bad), the time of the
last mapper line, and how many "Could not register" events there were.
IMPORTANT: use --since = the start time of the CURRENT job, otherwise the tail of a previous
(killed) job is counted as a model.  The mapper only logs while it registers images: a long silence
means a global bundle adjustment is running (normal, but on 1000+ images with 32k features each it can
take tens of minutes - see CameraSet_Training_Report_2026-10-19.md).
"""
import re
import subprocess
import sys

since = sys.argv[1] if len(sys.argv) > 1 else "-60 min"
out = subprocess.run(["journalctl", "-u", "gs-server", "--since", since, "--no-pager"],
                     capture_output=True, text=True).stdout
counts, last_ts = [], None
for line in out.splitlines():
    m = re.search(r"incremental_mapper\.cc:375\] Registering image #(\d+) \((\d+)\)", line)
    if m:
        counts.append(int(m.group(2)))
        last_ts = line[:15]
models, cur = [], 0
for c in counts:
    if c < cur:            # counter reset -> a new sub-model started
        models.append(cur)
    cur = c
if counts:
    models.append(cur)
print("since", since, "-> models (max registered each):", models, "| last mapper line:", last_ts)
print("could-not-register events:", sum(1 for l in out.splitlines() if "Could not register" in l))
