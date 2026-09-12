# GS Training Report: 12 New Scan Datasets

Date: 2026-08-13
Server: `http://136.64.198.146:8080`
Jobs: starting from `d90ce883-...` (first in the batch, `C_Deck_FWD_nocolmap`)

## 1. What Was Trained

12 folders from `D:\VSProjects\GS_Sever\data`, each trained two ways (without COLMAP and with COLMAP):

1. C_Deck_FWD (1814 photos)
2. D_Deck_FWD_Section 1 (Front Tip) (660 photos)
3. D_Deck_FWD_Section 2 (behind section 1) (583 photos)
4. SSUS_AFT_Engine_Room_ES22_iPhone_P3DGL (479 photos)
5. SSUS_AFT_Engine_Room_P2.1_02_iPad_P3DNW (543 photos)
6. TY_UpperDeckAFT_UseThisOne (1107 photos)
7. core machines next to long tube (192 photos)
8. core propellers 2 (68 photos)
9. engine room b deck (384 photos)
10. front tip full (1062 photos)
11. inside big towers (366 photos)
12. long silver tube (296 photos)

Total: 24 jobs (12 without COLMAP + 12 with COLMAP).

## 2. Without COLMAP — All 12 Successful

Point clouds downsampled from the original `.ply` files to ~2M points. Several datasets needed a scale check (mm vs m) — all 12 turned out to already be in meters, no correction needed (verified by matching the cloud's horizontal extent against the camera trajectory).

| Dataset | Job ID | Time | Final Splats |
|---|---|---|---|
| C_Deck_FWD | d90ce883-75db-4a5e-b769-9712957f4648 | 24 min | 2,004,543 |
| D_Deck_Section_1 | 02056964-f302-47f3-ba00-740c9bfc1d76 | 24 min | 1,989,221 |
| D_Deck_Section_2 | 54de76d2-dc01-4f35-8f56-5f7b091b7c9d | 23 min | 1,998,296 |
| SSUS_iPhone | 56bf62dd-e09e-4a11-87b1-e30c4c55ca45 | 25 min | 1,995,280 |
| SSUS_iPad | be2d94d8-2aff-4e44-9e6b-369d5c6b0fd4 | 25 min | 1,976,726 |
| TY_UpperDeckAFT | 9a181af7-f3aa-4d20-974e-bf575f4f4d95 | 24 min | 2,016,628 |
| core_machines | 2151e56b-2fc3-4659-acc3-c4768004e5dd | 27 min | 1,992,254 |
| core_propellers_2 | 6321bba9-52e1-4586-8846-b8d4943f63a2 | 31 min | 1,959,917 |
| engine_room_b_deck | dd055e8c-e802-46d0-8a72-39d6ba24cc85 | 22 min | 1,977,860 |
| front_tip_full | bee9c851-f202-4392-b95a-8016bac5d7ab | 23 min | 2,000,889 |
| inside_big_towers | 54520125-8bb6-4b37-b594-51321ab64d39 | 26 min | 1,976,605 |
| long_silver_tube | 07c03e9b-d0f8-4d76-8163-af2e9188116c | 30 min | 1,959,424 |

**All 12 are stable, ~2M splats each (matching the input point cloud size).** These are reliable results, usable right now.

### Direct Download Links (.ply) and Video Previews

Video previews are rendered by the server on demand (not pre-stored) - see section 12 of `CLIENT_TRAINING_MODES_DEVELOPER_GUIDE.md`. The links below are already rendered and working; if a specific one is unavailable (the server may not have kept it), just re-request it with `POST /jobs/{job_id}/preview`.

| Dataset | Model (.ply) | Preview (video) |
|---|---|---|
| C_Deck_FWD | http://136.64.198.146:8080/jobs/d90ce883-75db-4a5e-b769-9712957f4648/model | http://136.64.198.146:8080/jobs/d90ce883-75db-4a5e-b769-9712957f4648/preview/download |
| D_Deck_Section_1 | http://136.64.198.146:8080/jobs/02056964-f302-47f3-ba00-740c9bfc1d76/model | http://136.64.198.146:8080/jobs/02056964-f302-47f3-ba00-740c9bfc1d76/preview/download |
| D_Deck_Section_2 | http://136.64.198.146:8080/jobs/54de76d2-dc01-4f35-8f56-5f7b091b7c9d/model | http://136.64.198.146:8080/jobs/54de76d2-dc01-4f35-8f56-5f7b091b7c9d/preview/download |
| SSUS_iPhone | http://136.64.198.146:8080/jobs/56bf62dd-e09e-4a11-87b1-e30c4c55ca45/model | http://136.64.198.146:8080/jobs/56bf62dd-e09e-4a11-87b1-e30c4c55ca45/preview/download |
| SSUS_iPad | http://136.64.198.146:8080/jobs/be2d94d8-2aff-4e44-9e6b-369d5c6b0fd4/model | http://136.64.198.146:8080/jobs/be2d94d8-2aff-4e44-9e6b-369d5c6b0fd4/preview/download |
| TY_UpperDeckAFT | http://136.64.198.146:8080/jobs/9a181af7-f3aa-4d20-974e-bf575f4f4d95/model | http://136.64.198.146:8080/jobs/9a181af7-f3aa-4d20-974e-bf575f4f4d95/preview/download |
| core_machines | http://136.64.198.146:8080/jobs/2151e56b-2fc3-4659-acc3-c4768004e5dd/model | http://136.64.198.146:8080/jobs/2151e56b-2fc3-4659-acc3-c4768004e5dd/preview/download |
| core_propellers_2 | http://136.64.198.146:8080/jobs/6321bba9-52e1-4586-8846-b8d4943f63a2/model | http://136.64.198.146:8080/jobs/6321bba9-52e1-4586-8846-b8d4943f63a2/preview/download |
| engine_room_b_deck | http://136.64.198.146:8080/jobs/dd055e8c-e802-46d0-8a72-39d6ba24cc85/model | http://136.64.198.146:8080/jobs/dd055e8c-e802-46d0-8a72-39d6ba24cc85/preview/download |
| front_tip_full | http://136.64.198.146:8080/jobs/bee9c851-f202-4392-b95a-8016bac5d7ab/model | http://136.64.198.146:8080/jobs/bee9c851-f202-4392-b95a-8016bac5d7ab/preview/download |
| inside_big_towers | http://136.64.198.146:8080/jobs/54520125-8bb6-4b37-b594-51321ab64d39/model | http://136.64.198.146:8080/jobs/54520125-8bb6-4b37-b594-51321ab64d39/preview/download |
| long_silver_tube | http://136.64.198.146:8080/jobs/07c03e9b-d0f8-4d76-8163-af2e9188116c/model | http://136.64.198.146:8080/jobs/07c03e9b-d0f8-4d76-8163-af2e9188116c/preview/download |

Intermediate checkpoints (every 2500 iterations) — `http://136.64.198.146:8080/jobs/{job_id}/exports`.

## 3. With COLMAP — Fragmentation Problem Found and Diagnosed

### 3.1 First Attempt (sequential matcher, default overlap = 10)

| Dataset | Job ID | Status | Images Registered | Final Splats |
|---|---|---|---|---|
| C_Deck_FWD | 7e85701c-... | completed | **57 / 1814 (3%)** | 6,398 |
| D_Deck_Section_1 | d9c6ff65-... | completed | 81 / 660 (12%) | 11,125 |
| D_Deck_Section_2 | 6c0097bd-... | completed | 50 / 583 (9%) | 6,457 |
| SSUS_iPhone | 645c1289-... | completed | 238 / 479 (50%) | 161,216 |
| SSUS_iPad | b475bfdf-... | completed | 90 / 543 (17%) | 11,644 |
| TY_UpperDeckAFT | 1c9736d1-... | **crashed** (Brush error) | 120 / 1107 (11%) | 0 |
| core_machines | a44e3232-... | completed | **192 / 192 (100%)** ✅ | 138,543 |
| core_propellers_2 | 8c872a47-... | completed | 41 / 68 (60%) | 20,678 |
| engine_room_b_deck | df7303de-... | completed | 123 / 384 (32%) | 25,240 |
| front_tip_full | 9af738a3-... | completed | 157 / 1062 (15%) | 18,380 |
| inside_big_towers | ce743f9a-... | completed | 57 / 366 (16%) | 81,066 |
| long_silver_tube | 79083d73-... | completed | 208 / 296 (70%) | 133,365 |

**Problem:** 11 of 12 jobs formally reported "completed," but in reality COLMAP failed to connect most photos into a single model — the incremental mapper split each scene into dozens of disconnected sub-models (e.g. C_Deck_FWD fragmented into 17 pieces, TY_UpperDeckAFT into 41), and Brush trained only on the largest fragment, ignoring the rest of the photos. Consequences:
- Suspiciously fast completion (11–80 minutes instead of the expected hours)
- Extremely low splat counts almost everywhere (except `core_machines`, where COLMAP worked as intended — 100% of images registered)
- 1 job (TY_UpperDeckAFT) crashed outright while loading the dataset into Brush — apparently a side effect of that same fragmentation (41 fragments instead of one model)

Working model for `core_machines` (the only reliable result from the v1 COLMAP batch): http://136.64.198.146:8080/jobs/a44e3232-9db3-4d34-aff9-94513033aaad/model
Video preview: http://136.64.198.146:8080/jobs/a44e3232-9db3-4d34-aff9-94513033aaad/preview/download

**Initial hypothesis:** COLMAP's default neighbor-search window for the sequential matcher (`--SequentialMatching.overlap`, default 10 frames) is too narrow for these captures.

### 3.2 Attempted Fix — overlap=50 (Did Not Help)

Added `sequential_overlap` to `colmap_runner.py` as a configurable parameter, re-ran jobs with `sequential_overlap=50` instead of 10. The result was only marginally better:
- C_Deck_FWD: 3% → 6% (57 → 107 of 1814)
- D_Deck_Section_1: 12% → 14% (81 → 92 of 660)

This means the issue isn't the sequential matcher's window width. Cancelled the remaining 9 planned re-runs pending root-cause diagnosis — no point burning hours of compute on a parameter that doesn't fix the problem.

### 3.3 Diagnosis — Exhaustive Matcher Test

Ran a test job on `D_Deck_FWD_Section 2` (583 photos, exhaustive matcher — all pairs instead of sequential neighbors, to test whether the pairing strategy itself was the issue). Result: exhaustive connected only **150/583 (26%)** — better than sequential's 9%, but still far from full connectivity.

To get a definitive answer rather than guess from indirect signals, queried COLMAP's database directly (`database.db`, table `two_view_geometries`, which stores the geometrically verified matches between every image pair):

- Total pairs checked (exhaustive = every possible pair): **169,653**
- Pairs with **zero** verified matches: **166,023 (97.9%)**
- Pairs with ≥15 matches (COLMAP's threshold for a "reliable" pair): only **3,630 (2.1%)**
- Median matches per pair: **0**

For comparison — the same metric for `core_machines` (the only dataset where COLMAP worked normally, 100% registration):
- Pairs with ≥15 matches: **1,186 of 2,211 (53.6%)** — 25x the rate of D_Deck_Section_2.

**Conclusion:** the problem isn't matcher configuration (neither the sequential window nor exhaustive matching fixes it) — most of these scenes (apparently due to reflective/repetitive metal surfaces typical of a ship engine room/deck) simply lack enough stable, distinguishable SIFT features between frames for reliable matching. This is a limitation of the photographs themselves, not a tunable parameter.

## 4. Pivot: "Known Poses" Hybrid Mode + COLMAP Triangulation

### 4.1 The Idea

COLMAP can't **estimate** camera poses from poorly-matching photos — but we don't need it to: we already have accurate poses from the scanner's own LiDAR/IMU tracking (the same ones used in the no-COLMAP mode). Instead of asking COLMAP to derive poses (`mapper`, which fails), we feed it the known poses directly and ask it only to **triangulate 3D points** from whatever matches it can find (`point_triangulator`) — which doesn't require a fully connected image graph, just 2+ observations per point. Points are then lightly refined via `bundle_adjuster` and merged with the original LiDAR cloud — yielding a denser, more complete initialization than either source alone, with no coordinate-frame alignment issues (COLMAP uses the same poses as LiDAR, not its own independently-derived frame).

Implemented as a new `training_mode = hybrid_known_poses` on the server (`gs_server/known_pose_triangulation.py`, ~400 lines):
1. Recover the original (OpenCV/COLMAP-convention) camera poses from `transforms.json`.
2. `feature_extractor` + `matcher` — same as before.
3. Write a "known pose" sparse model (`cameras.txt`/`images.txt`/empty `points3D.txt`) and run `point_triangulator` (matches by filename, not by ID).
4. `bundle_adjuster` — light refinement.
5. Export triangulated points, merge with the original `points3d.ply`.
6. Train Brush on the merged dataset (reuses the already-proven no-COLMAP code path).

### 4.2 Three Bugs Found and Fixed During Testing

**Bug 1 — spaces in filenames.** COLMAP's own `images.txt` parser splits on whitespace for every field, including the filename — and many of our folders have spaces in their photo names (e.g. `core propellers 2_SensorFrame_00000.jpg`, inherited from the folder name). `point_triangulator` crashed with "Image with name core does not exist in database" (it truncated the name to the first word). Fixed: COLMAP's internal steps now use symlinks with safe names (`img_00001.jpg` etc.); the source dataset itself is untouched.

**Bug 2 — "confidently wrong" triangulated points.** After the first successful run on `core_propellers_2`, the intermediate model appeared "doubled." Investigation showed: the triangulated COLMAP point cloud had a bounding box reaching X=5938 at a scene scale of ~20 units! Cause — a classic degenerate triangulation case: with nearly-parallel viewing rays (a weak baseline, characteristic of exactly these poorly-matching scenes), a small angular error produces a huge depth error, while the pixel reprojection error stays small — so filtering on COLMAP's standard error metric doesn't catch these points. Fixed: added a sanity filter based on the scene's reasonable bounds (the original LiDAR cloud's bbox + 50% margin) — discards "flown-away" points before merging.

**Bug 3 — camera-pose refinement in `bundle_adjuster` was pulling part of the scene "sideways."** After fixing bug 2, ran another test — the intermediate model still looked doubled, with one copy rotated relative to the other. Checked `bundle_adjuster --help`: by default it has `--BundleAdjustment.refine_extrinsics=1` — meaning it doesn't just refine 3D points, it **freely moves camera poses** too. For cameras with weak/sparse matches (typical for these scenes), this lets them drift into a different, locally self-consistent but geometrically wrong solution — taking their triangulated points with them. Fixed by locking every camera parameter (`refine_extrinsics=0`, `refine_focal_length=0`, `refine_principal_point=0`, `refine_extra_params=0`) — `bundle_adjuster` now only refines point positions; camera poses stay exactly as known from LiDAR tracking.

Verified directly: for 300 random COLMAP points, computed distance to the nearest LiDAR point — median 0.042 at a scene scale of ~20 units, 98% within 0.3. Geometrically correct, no more doubling in the data.

### 4.3 Clarification: the Doubling Was, and Remains, Not in This Code

After fixing bug 3, ran one more test — the intermediate render still showed visible doubling. Before continuing to chase what might not be a bug, asked for a comparison against the plain no-COLMAP mode's result on the same dataset (job `6321bba9`, predating the hybrid mode entirely) — **the same effect is present there too**. This confirms: the doubling is a property of the source data/poses specific to `core_propellers_2` (the folder name is plural — the scan may well physically contain two propellers), not a defect in the hybrid algorithm. Three real bugs (filename spaces, flown-away triangulation points, unconstrained pose refinement) were found, fixed, and verified independently; the known-pose triangulation algorithm itself works correctly.

### 4.4 Test on core_propellers_2 (After All Fixes)

- Merged cloud: 1,959,916 LiDAR points + 32,495 (post-filter, from 34,500 raw) COLMAP points = 1,992,411.
- COLMAP points' bounding box post-filter: essentially matches the LiDAR cloud (within the expected margin).
- Brush training completed with no splat collapse, stable splat count.

### 4.5 Mass Rollout (Resumed)

Queued the remaining 10 datasets (all except `core_machines`, which was already fine) with `matcher_type=exhaustive` (maximum point coverage, since connectivity is no longer critical for this pipeline). Matching-time estimate (the dominant cost, scaling quadratically with image count):

| Dataset | Photos | Estimated Matching Time |
|---|---|---|
| C_Deck_FWD | 1814 | ~19h |
| TY_UpperDeckAFT | 1107 | ~7h |
| front_tip_full | 1062 | ~6.5h |
| D_Deck_Section_1 | 660 | ~2.5h |
| D_Deck_Section_2 | 583 | ~2h |
| SSUS_iPad | 543 | ~1.7h |
| SSUS_iPhone | 479 | ~1.3h |
| engine_room_b_deck | 384 | ~0.85h |
| inside_big_towers | 366 | ~0.8h |
| long_silver_tube | 296 | ~0.5h |

**Total ≈ 42 hours** — but unlike the previous approach (hours on the incremental mapper with no guarantee of a result), this is a predictable, bounded cost — the rest of the pipeline (extraction, triangulation, bundle adjustment, merging) is fast.

All 10 datasets were re-queued (after the bug-3 fix) in ascending order of size.

## 5. Point Cloud Quality Improvement: Denoising + COLMAP Priority

After confirming the hybrid algorithm works correctly, a follow-up request came in: now that COLMAP points are reliable, can we reduce the influence of noise in the raw LiDAR cloud and shift the balance toward the more accurate COLMAP data where it exists?

### 5.1 Implementation

Added two new steps to the cloud-merging stage (`known_pose_triangulation.py`), both built on a scipy KD-tree (installed `scipy` on the server as a separate package):

1. **Statistical outlier removal (SOR) on the LiDAR cloud.** For each point, compute the mean distance to its 12 nearest neighbors; points whose mean distance is abnormally large relative to the cloud-wide average (threshold: mean + 2.5×std) are discarded. This is typical scanner noise / multipath reflections off the same metal surfaces that also hurt COLMAP.
2. **Local COLMAP priority.** Any LiDAR point within some radius of a COLMAP point is removed, so the more accurate photogrammetric point "wins" locally. LiDAR remains the sole source everywhere COLMAP has no coverage.

### 5.2 Bug 4 — the Fixed Priority Radius Was Uncalibrated

The first run with this code (`core_propellers_2`, radius = 0.05, hardcoded) cut the original cloud from 1,959,916 down to 1,200,795 points — a 38.7% loss, noticeably more than expected from "just cleaning up noise." Investigation: the SOR step alone removed only 1.4% (1,959,916 → 1,932,502) — that's normal. The rest of the loss came from the COLMAP-priority step: the mean spacing between neighboring LiDAR points in this cloud is only ~0.0109 units, so a fixed radius of 0.05 (~5x the cloud's own density) wiped out a ~25-point neighborhood around every COLMAP point instead of simply ceding that point's own immediate spot.

**Fix:** the priority radius is no longer a fixed number — it's now computed automatically as 1.5x the mean nearest-neighbor distance of the specific cloud being processed, adapting to each dataset's density rather than one hardcoded scale.

### 5.3 Result After Calibration (core_propellers_2)

| Stage | Points | % of Original |
|---|---|---|
| Original LiDAR cloud | 1,959,916 | 100% |
| After SOR (noise removed) | 1,932,502 | 98.6% |
| After COLMAP priority (auto radius = 0.0164) | 1,811,901 | 92.5% |
| + COLMAP points | +33,051 | — |
| **Final merged cloud** | **1,844,952** | — |

The balance is now sensible: removed a noticeable but not excessive amount of noise (1.4%), ceded a modest area proportional to COLMAP's own point count (6.1%), without losing the bulk of LiDAR coverage.

### 5.4 Mass Rollout (Restarted with the Improved Pipeline)

All 10 datasets were re-queued on top of the fixed version — denoising + calibrated COLMAP priority. In progress.

## 6. Status as of This Writing

- **No-COLMAP: 12/12 done, all results reliable**, download links in section 2.
- **Pure COLMAP (without known poses): not viable** for 10 of 12 scenes due to insufficient texture — not a path forward for this data.
- **Hybrid mode (known poses + COLMAP triangulation + LiDAR denoising): algorithm implemented and verified correct.** Four bugs found (filename spaces, flown-away triangulation points, unconstrained pose refinement in bundle adjustment, uncalibrated COLMAP-priority radius) have been fixed and verified; the visual doubling observed on one test dataset (`core_propellers_2`) was traced to a property of the source data, not an algorithm defect. The mass rollout across all 10 remaining datasets is running on the final pipeline version; this report will be updated with results and links as they complete.

## 7. Brush Regularization Calibration (opac/scale decay)

After the hybrid mode work wrapped up, a new issue surfaced: the edges of objects in finished GS models showed a lot of noisy, elongated "needle" Gaussians ("rays" sticking out from silhouette edges).

### 7.1 A real bug found: regularization never actually reached Brush

Inspecting `brush_runner.py` showed that `opac_decay`/`scale_decay` were read from config but **never forwarded** into the actual Brush launch command — the command was built without them. Every model trained up to that point had silently used Brush's own defaults (`opac-loss-weight=1e-9`, `scale-loss-weight=1e-8`), not the configured values. On top of that, the old config values (0.007 / 0.004) turned out to be in the wrong units entirely — roughly 7 orders of magnitude off from Brush's real flags.

Fixed: added the real flags (`--opac-loss-weight`, `--scale-loss-weight`) to the launch command, with values corrected to the right units — set to 100x Brush's defaults as a starting point for calibration.

### 7.2 A calibration sweep showed this lever doesn't solve the problem

Tested 3x, 10x, 30x, 100x of Brush's defaults in sequence on a fast test dataset (`core_propellers_2`), tracking not just splat count but anisotropy (max/min Gaussian scale ratio — higher means more "needle-like"):

| Multiplier | Splats | % of baseline | Median anisotropy |
|---|---|---|---|
| 1x (previous default) | 1,844,954 | — | 391 |
| 3x | 1,849,760 | ~0% | 521 |
| 10x | 1,842,245 | ~0% | 709 |
| 30x | 420,126 | −77% | 719 |
| 100x | 113,904 | −94% | 508 |

**Finding:** stronger regularization does not make surviving Gaussians less needle-like — median anisotropy only rises (391 → 719). What actually happens is a blunt, untargeted cut in splat count: the more that gets pruned, the larger the survivors have to grow to cover the resulting gaps in the scene (the 99th-percentile max Gaussian size grew nearly 8x at 100x). That's what visually read as "the object became transparent and blurry" at 100x — not reduced noise.

**Resolution:** reverted regularization to Brush's own defaults (`opac_decay=1e-9`, `scale_decay=1e-8`). The silhouette-needle problem needs a different fix.

## 8. Chasing Needle Artifacts at Object Edges

Since loss-based regularization didn't help, several other approaches were tried — post-processing and training-time densification parameters.

### 8.1 A shape-based post-processing filter didn't fix the visible artifact

Wrote `gs_server/floater_filter.py`: removes Gaussians that are both highly elongated (anisotropy above a threshold) and large in absolute scene terms. A screenshot you shared revealed the visible artifact wasn't thin needles at all — it was wide, bright glare (semi-transparent, only mildly elongated, but large "pancake" blobs). Even the aggressive filter variant (removing ~10% of splats) left the glare in the screenshot untouched.

Switched to a pure size cutoff (no elongation requirement) — that fully removed the glare. But the same screenshot, from a different angle, then showed thin needle spikes still present right along the object's silhouette edge — they'd simply been hidden behind the glare from the large blobs before.

### 8.2 Needle vs. ordinary surface: shape alone isn't a sufficient signal

A direct attempt to filter the remaining thin elongated splats (anisotropy alone, at a smaller scale) showed that criterion catches 20–46% of the *entire* model — because ordinary smooth surfaces in 3D Gaussian Splatting are themselves represented by thin, mildly-elongated splats. Anisotropy alone can't distinguish a "needle artifact" from a "normal flat surface patch."

Added a second signal: distance from each Gaussian's center to the nearest point in the source scan cloud (LiDAR+COLMAP). The idea: a needle grown into empty space during densification should be both elongated and detached from real scanned geometry; a normal surface splat is elongated but close to the scan. Combining both checks cut the affected fraction to 0.4–8% depending on threshold — but per your visual check, it had almost no effect on the visible spikes.

### 8.3 Checked training-time densification parameters — also no effect

Next hypothesis: if post-processing can't find the right population, maybe the spikes form during training itself (densification/growth), and the fix is to simply not let them grow. Tested earlier growth cutoff and stricter growth thresholds across three variants — key finding: **final splat count barely differed from the size of the input point cloud** regardless of growth settings (1,845,370 / 1,851,368 / 1,849,739 vs. a baseline of 1,844,954). This means Brush in hybrid mode adds almost no new splats on top of the initial point-cloud seeding — so controlling growth during training is pointless here; the splat population is already almost entirely determined by the input cloud.

**Current status:** the exact nature of the edge spikes remains unresolved. Working hypothesis: these are photometric artifacts (splats that form to explain blended/semi-transparent pixels at the object/background boundary in source photos), not geometric noise and not a training artifact — an open question for further work.

### 8.4 Found a real source of noise: weakly-constrained COLMAP points

While debugging the COLMAP crash (see section 9), also checked the quality of the triangulated points the hybrid mode blends in on top of the LiDAR cloud. It turned out **half** of the triangulated COLMAP points had a track of just 2 observing images — the minimum possible, and extremely sensitive to small angular error (typical right at object edges, where viewing angles are nearly parallel). These points were previously filtered only by overall scene bounding box, nothing else.

Added a track-length filter (minimum 3 observing images by default) — COLMAP computes this field itself, it just wasn't being used. Applied automatically to all new hybrid runs going forward.

### 8.5 A bug in the filter itself: an absolute size cutoff broke on sparse models

While reviewing results, you flagged: "the filter at the end is too strong, a large part of the model is lost, especially with COLMAP-only." Investigation confirmed it: on a model trained in pure COLMAP mode (no LiDAR, sparse initialization), the filter removed **59% of the entire model** (12,922 → 5,290 splats), even though the same filter cleanly removed <1% on dense LiDAR-initialized models.

Root cause: the absolute size cutoff (10cm) was calibrated against one reference scene with dense LiDAR initialization, where a typical splat is a fraction of a centimeter. On a sparsely-initialized COLMAP-only model, the typical splat is naturally larger (fewer points, each covering more area), and 10cm turned out to be *below* that model's own median splat size — so the filter was cutting "normal" splats as if they were outliers.

**Fix:** the threshold is now relative — not a fixed number of centimeters, but a multiplier (15x) of that specific model's own median splat size. Retroactively re-ran the filter on all already-trained models without retraining:

| Model | Before (absolute cutoff) | After (relative) |
|---|---|---|
| COLMAP-only (see section 11) | −59% | **−1.13%** |
| Other 14 hybrid/LiDAR models | −0.2…0.6% | −0.26…0.61% (unchanged, as expected) |

## 9. COLMAP Crash on "Disconnected" Frames — Found and Fixed

When resuming the mass rollout across all 11 datasets (the 10 from the earlier batch plus one that had never been queued at all — `core_machines_next_to_long_tube`) — 6 of 11 crashed with a COLMAP failure (`point_triangulator`, `std::out_of_range` in `ObservationManager`, SIGABRT).

Reproduced the crash manually and found the exact cause in COLMAP's own log: `Loading images... 192 in 1.392s (connected 191)` — one of the 192 frames had zero verified matches to any other frame at all (a fully isolated node in the match graph). `point_triangulator` can't handle such a frame gracefully and crashes instead of simply skipping it.

**Fix:** before triangulation, the COLMAP database (`two_view_geometries` table) is now checked for fully disconnected frames, and any found are excluded from the known-pose model fed to `point_triangulator`. This doesn't affect final training — Brush uses the full frame set and poses from `transforms.json` regardless of this step; only COLMAP's point triangulation for that one frame (which had no matches to contribute anyway) is affected.

All 6 crashed datasets were recreated and successfully retrained after the fix.

## 10. Final Mass Rollout Results — All 11 Datasets

All datasets were trained in hybrid mode (`hybrid_known_poses`) on the final pipeline version: denoising + calibrated COLMAP priority + track-length filter + Brush defaults for regularization + relative post-processing filter. Splat counts are from the actual downloadable file (post-filter), not the intermediate progress report.

| Dataset | Job ID | Final Splats | Model Link | Preview (video) |
|---|---|---|---|---|
| C_Deck_FWD | cc34ee98-98c8-4c3d-a28b-e455805b6f57 | 1,747,709 | http://136.64.198.146:8080/jobs/cc34ee98-98c8-4c3d-a28b-e455805b6f57/model | http://136.64.198.146:8080/jobs/cc34ee98-98c8-4c3d-a28b-e455805b6f57/preview/download |
| D_Deck_FWD_Section_1_Front_Tip | 25be7ece-eeb5-43f0-97db-a40201271834 | 1,863,815 | http://136.64.198.146:8080/jobs/25be7ece-eeb5-43f0-97db-a40201271834/model | http://136.64.198.146:8080/jobs/25be7ece-eeb5-43f0-97db-a40201271834/preview/download |
| D_Deck_FWD_Section_2_behind_section_1 | 1bb10401-3235-4980-b4a2-ab70b8d31d3c | 1,891,234 | http://136.64.198.146:8080/jobs/1bb10401-3235-4980-b4a2-ab70b8d31d3c/model | http://136.64.198.146:8080/jobs/1bb10401-3235-4980-b4a2-ab70b8d31d3c/preview/download |
| SSUS_AFT_Engine_Room_ES22_iPhone_P3DGL | 790ef5c3-1ebd-472a-8aaf-f6eb57ff65f7 | 1,650,321 | http://136.64.198.146:8080/jobs/790ef5c3-1ebd-472a-8aaf-f6eb57ff65f7/model | http://136.64.198.146:8080/jobs/790ef5c3-1ebd-472a-8aaf-f6eb57ff65f7/preview/download |
| SSUS_AFT_Engine_Room_P2.1_02_iPad_P3DNW | 8e7a0f8f-bc96-4fc8-931c-2683d199c22e | 1,762,234 | http://136.64.198.146:8080/jobs/8e7a0f8f-bc96-4fc8-931c-2683d199c22e/model | http://136.64.198.146:8080/jobs/8e7a0f8f-bc96-4fc8-931c-2683d199c22e/preview/download |
| TY_UpperDeckAFT_UseThisOne | ad1b5983-cbaf-4e32-a998-d6c012f5781c | 1,542,136 | http://136.64.198.146:8080/jobs/ad1b5983-cbaf-4e32-a998-d6c012f5781c/model | http://136.64.198.146:8080/jobs/ad1b5983-cbaf-4e32-a998-d6c012f5781c/preview/download |
| core_machines_next_to_long_tube | 53676ac9-379d-41af-8d31-d91482e7aebf | 1,638,948 | http://136.64.198.146:8080/jobs/53676ac9-379d-41af-8d31-d91482e7aebf/model | http://136.64.198.146:8080/jobs/53676ac9-379d-41af-8d31-d91482e7aebf/preview/download |
| engine_room_b_deck | d507e3a3-29d4-4771-9bc7-1ddb183ec821 | 1,880,573 | http://136.64.198.146:8080/jobs/d507e3a3-29d4-4771-9bc7-1ddb183ec821/model | http://136.64.198.146:8080/jobs/d507e3a3-29d4-4771-9bc7-1ddb183ec821/preview/download |
| front_tip_full | bb052a50-a579-45fb-b338-a4f7c94fd1ab | 1,741,954 | http://136.64.198.146:8080/jobs/bb052a50-a579-45fb-b338-a4f7c94fd1ab/model | http://136.64.198.146:8080/jobs/bb052a50-a579-45fb-b338-a4f7c94fd1ab/preview/download |
| inside_big_towers | ad381047-3de3-47d1-90a1-fa10de8a614b | 1,678,151 | http://136.64.198.146:8080/jobs/ad381047-3de3-47d1-90a1-fa10de8a614b/model | http://136.64.198.146:8080/jobs/ad381047-3de3-47d1-90a1-fa10de8a614b/preview/download |
| long_silver_tube | e8098869-814a-42e6-9cf0-0458cb4eedf7 | 1,620,657 | http://136.64.198.146:8080/jobs/e8098869-814a-42e6-9cf0-0458cb4eedf7/model | http://136.64.198.146:8080/jobs/e8098869-814a-42e6-9cf0-0458cb4eedf7/preview/download |

Intermediate checkpoints and the unfiltered version — `http://136.64.198.146:8080/jobs/{job_id}/exports`.

## 11. Pilot: A New Data Source — Phone + External LiDAR

A separate request: train on a new data type — a smartphone with an external LiDAR module attached, capturing alongside the camera. Unlike the earlier datasets, the external LiDAR points are collected **raw, frame by frame** (not yet a point cloud — unprocessed spherical measurements with no pose), and the phone's pose comes from ARKit, which was explicitly flagged as not fully trustworthy. The external LiDAR's axes and scale may not match the phone's.

Test dataset: `GS_20260805_105828` (114 camera frames, 111 LiDAR frames).

### 11.1 A New External-LiDAR Calibration Pipeline

Wrote `external_lidar_to_pointcloud.py`:
1. Real COLMAP SfM on the images — independently trustworthy camera poses, but in an arbitrary (non-metric) scale.
2. Metric scale recovery — comparing relative inter-frame motion between COLMAP and ARKit (ARKit's absolute pose isn't trusted, but its relative motion over short intervals is usually reasonably metric).
3. Converting raw spherical LiDAR measurements (range + two angles) to Cartesian coordinates, placing each LiDAR frame in world space using the camera pose interpolated (SLERP) to that same timestamp.
4. ICP calibration of the unknown rigid camera↔LiDAR offset — registering the aggregated LiDAR cloud against COLMAP's sparse points.
5. Exporting the calibrated point cloud + a `transforms.json` with metric poses.

**Found and fixed a format bug along the way:** this source's `transforms.json` stores `transform_matrix` column-major (ARKit/Swift's native layout), not row-major like everywhere else in the pipeline — as a result, camera translation was being read as a zero vector. Also found and fixed missing per-frame camera intrinsics (they were only present at the top level of the JSON, while the rest of the pipeline expects them per-frame).

Calibration outcome: COLMAP registered 114/114 frames, scale factor 0.7998, ICP found a ~66° rotation and ~0.89 translation between camera and LiDAR (expected for a separate accessory), final residual ~1.7% of the scene diagonal. Visual inspection (overlaying the LiDAR cloud on COLMAP's points) confirmed dense overlap with no gross misalignment; a noticeable "spiral" blur remains around the core — likely because a single global correction doesn't fully compensate for the camera↔LiDAR offset if it varied during capture (an open question, doesn't block use).

### 11.2 Results — Three Modes on One Dataset

| Mode | Job ID | Data | Final Splats | Model Link | Preview (video) |
|---|---|---|---|---|---|
| With COLMAP (images only) | b7638369-62a5-4109-9165-597cfa1bf5cc | images | 12,776 | http://136.64.198.146:8080/jobs/b7638369-62a5-4109-9165-597cfa1bf5cc/model | http://136.64.198.146:8080/jobs/b7638369-62a5-4109-9165-597cfa1bf5cc/preview/download |
| Without COLMAP (calibrated LiDAR) | 0744429c-118e-41dd-a740-3c2b64513382 | external LiDAR | 2,371,591 | http://136.64.198.146:8080/jobs/0744429c-118e-41dd-a740-3c2b64513382/model | http://136.64.198.146:8080/jobs/0744429c-118e-41dd-a740-3c2b64513382/preview/download |
| Hybrid | 209dd64d-eefe-4564-a35d-5f771868f6b8 | LiDAR + COLMAP | 2,277,282 | http://136.64.198.146:8080/jobs/209dd64d-eefe-4564-a35d-5f771868f6b8/model | http://136.64.198.146:8080/jobs/209dd64d-eefe-4564-a35d-5f771868f6b8/preview/download |

**"COLMAP-only" mode produced noticeably fewer splats (12,776 vs. ~2.3–2.4M for the other two).** Investigation showed COLMAP's incremental mapper fragmented the reconstruction into 4 disconnected pieces (0/1/2/3), and training ran on only the largest one — the same category of problem as section 3 (insufficient texture/connectivity between frames for pure SfM). The LiDAR-based modes (without-COLMAP and hybrid) don't have this problem, since they don't depend on image-graph connectivity.

## 12. Summary

- **12 original + 11 new datasets (23 total) trained in hybrid mode, all results stable and available via direct links** (sections 2 and 10).
- Hybrid mode is now the standard approach for low-texture scenes (metal surfaces, engine rooms) and for new data types (external LiDAR), where pure COLMAP regularly fragments.
- Brush regularization was reverted to defaults after a calibration sweep showed it doesn't reduce needle-like anisotropy — it just crudely and ineffectively thins the model.
- Silhouette-edge needle artifacts are partially addressed (glare from large semi-transparent blobs removed, contribution from weakly-constrained COLMAP points trimmed) but not fully solved; working hypothesis is a photometric effect from blended edge pixels in source photos — an open question for further work.
- The post-processing filter was switched to a relative (self-calibrating) threshold after the absolute threshold proved unsuitable for sparse (COLMAP-only) reconstructions.
- Found and fixed a COLMAP crash on fully disconnected (isolated) frames — affected 6 of 11 datasets in the latest batch.
- Implemented and verified a calibration pipeline for a new data source — a smartphone with an external LiDAR module; includes a fix for the ARKit matrix format and ICP calibration of the unknown camera↔LiDAR offset.

## 13. Visual comparison of training variants (photos + flythrough videos)

**All material:** [Google Drive — splat_compare](https://drive.google.com/drive/folders/1e3rcvwtjEpxFaVE6AGqCsdWoiNX1zL6c?usp=sharing)

To visually check quality across training modes for each of the 12 scenes, the models (`models/*.ply`) were rendered from the same viewpoints the source photos were taken from. The Drive folder has one subfolder per scene, each containing:

- `comparison_grid.png` — a grid of 4 viewpoints × model variants: real photo | COLMAP only (only for `core_machines_next_to_long_tube` — the one scene where pure COLMAP didn't fragment, see section 3) | Hybrid | **Hybrid (no filter)** | Without COLMAP.
- `compare_flythrough.mp4` — a smooth flythrough video along an interpolated (slerp/lerp) path through the real capture poses, variants shown side by side.

The **Hybrid (no filter)** column is `model_30000.ply` before the post-processing floater filter (section 8), downloaded directly from the server separately from the already-filtered version. Added per your comment that the filter sometimes removes too much: the comparison shows that for these (dense, LiDAR-initialized) hybrid models the filter removes relatively little (~0.3–0.9% of splats, consistent with section 8.5) — mostly a handful of bright, colorful stray-blob outliers visible from certain angles (e.g. frame 191 of `core_machines_next_to_long_tube`); on most other viewpoints the difference is minimal.

The renderer is a custom approximate one (GPU/torch disc-splatting with depth-sorted alpha compositing), not an exact copy of Brush's shader — it ignores the view-dependent part of the SH coefficients, so glare/reflections are rendered more coarsely than in Brush's actual viewer. That's sufficient for comparing quality across training variants, though not for absolute photorealism.
