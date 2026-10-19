#!/usr/bin/env python3
"""Run ON the server. Read-only quality check of a COLMAP database's matching.

  sudo python3 match_stats.py /data/jobs/<job_id>/colmap/database.db

Answers "did matching work?" independently of the mapper: how many image pairs were verified, how many
consecutive frames (in file-name order) are connected, and their inlier counts.  Healthy video sets have
~all consecutive pairs verified with thousands of inliers.  If this looks good but the mapper still
produces tiny models, the problem is triangulation (too few features / too little baseline), not matching.
"""
import collections
import sqlite3
import statistics
import sys

db = sys.argv[1]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
names = dict(c.execute("SELECT image_id, name FROM images"))
pairs = c.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0").fetchall()
print(f"images={len(names)}  pairs_in_table={c.execute('SELECT COUNT(*) FROM two_view_geometries').fetchone()[0]}"
      f"  verified_pairs={len(pairs)}")
M = 2147483647
ids = [k for k, _ in sorted(names.items(), key=lambda kv: kv[1])]
pos = {k: p for p, k in enumerate(ids)}


def split(pid):
    j = pid % M
    return (pid - j) // M, j


gaps = collections.Counter(abs(pos[split(p)[0]] - pos[split(p)[1]]) for p, _ in pairs)
print("verified pairs by index distance:", {g: gaps[g] for g in (1, 2, 3, 5, 10, 20, 30)})
inl = [r for p, r in pairs if abs(pos[split(p)[0]] - pos[split(p)[1]]) == 1]
print(f"consecutive pairs verified: {len(inl)}/{len(ids) - 1}")
if inl:
    print("inliers of consecutive pairs: median", statistics.median(inl), "p10", sorted(inl)[len(inl) // 10], "min", min(inl))
kp = [r[0] for r in c.execute("SELECT rows FROM keypoints")]
print("keypoints per image: median", statistics.median(kp))
