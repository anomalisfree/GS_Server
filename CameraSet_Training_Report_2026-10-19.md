# CameraSet: can a Gaussian Splatting model be trained from this data?

Date: 2026-10-19 (experiments ran 2026-10-12 .. 2026-10-15)
Server: `gs-training-server` (GCP, project `dgs-training`, NVIDIA L4)
Data: `data/CameraSet/` (1.3 GB, **git-ignored** - copy it by hand, see `CONTINUE_ON_ANOTHER_COMPUTER.md`)
Audience: internal, for continuing the work from another machine

## 0. Verdict

**Yes - but only from the two videos.** The `.ply` and `.dp` files cannot train a Gaussian Splatting model on
their own (no photographs, no camera poses). Five trainings were run on the server; all produced usable,
recognisable models of the street scene (sidewalk, hedges, bushes, parked cars), with the usual 3DGS
weaknesses: empty/black sky and far field and a few bright streaks where far geometry is badly constrained.
There is **no quantitative quality metric** (no held-out frames, PSNR/SSIM are `None`); all judgements below
are visual, on rendered flythroughs.

## 1. What is in `data/CameraSet/`

Capture date 2026-09-14, a residential street in California, "3 camera" rig experiments with a Dot3D
scanner app on an iPhone 12 Pro Max-class device.

| Files | What they are | Usable for GS? |
|---|---|---|
| 2 x `.MOV` ("Scan 01" 125.1 s / 3754 frames, "Scan 02" 66.0 s / 1979 frames) | 1080x1920 portrait, 30 fps, sharp (Laplacian variance median 697 / 1111) walking videos | **Yes** - source of every model below |
| 6 x `.ply` (1.3 M .. 29.3 M points) | Coloured point clouds, `x y z r g b` only. Extent ~8-27 m. No normals, no cameras | No images/poses, so no. **Do not** feed them to `hybrid_colmap_pointcloud`: the server copies the cloud as-is, it does not align it to COLMAP's frame |
| 6 x `.dp` | Proprietary **Dot3D 6.4.2** project files (`DP_BINARY_DATA__`, iPhone13,4 / iOS 18.7.8). Contain a 320x240 thumbnail, 20x15 icons and an opaque binary blob | No - no photos, poses not extractable without the vendor SDK |

Consequence for `hybrid_known_poses` (the best mode for LiDAR scans): not possible from this data. It needs
per-frame poses + intrinsics in a Nerfstudio `transforms.json`. Ask the scanner vendor for an export of
posed images (or poses + timestamps to sync with the videos) if that is wanted.

## 2. Experiments

All jobs: `single_camera: true`, Brush 30000 steps, `max_resolution 1920`. Frames = the sharpest frame of every
N-th video frame window (`tools/client/video_to_frames.py`). Job ids are the server's 8-character prefixes.

| Job (id) | Frames uploaded | COLMAP setup | Registered into the trained model | Splats | Result |
|---|---|---|---|---|---|
| `CameraSet_Scan02_video` (`37c38b8a`) | 197 (every 10th) | exhaustive, 32k feat. | 149 (76 %); other fragments 18/13/2 | 77,747 | OK, near field good |
| `CameraSet_Scan01_video` (`81824c4b`) | 375 (every 10th) | exhaustive | 245 (65 %); another 121-image model discarded | 95,521 | OK, sidewalk shadows very good |
| `CameraSet_Combined_Scan01_Scan02` (`92c05cc4`) | 572 (both videos) | exhaustive | 244 (43 %) = 127 Scan01 + 117 Scan02; a 189-image model (also mixed) discarded; 8 models in total | 143,548 | **Best coverage**; proves the two videos link in COLMAP (same place) |
| dense5 (1145 frames, every 5th) | 1145 | sequential + loop detection, 32k feat. | mapper reached only 239 -> 252 images in 1 h | - | **Abandoned**, see 3.2 |
| `CameraSet_Combined_dense8_light` (`7ceacdae`) | 716 (every 8th) | sequential overlap 20 + loop detection, **`max_image_size 1280`** (~12k feat.) | 231 | 61,320 | OK; whole job 2 h 01 min |
| `CameraSet_dense8_skymask` (`fc7ec144`) | same 231 undistorted images | `pointcloud_only` on a copy of the dense8_light set + sky masks | 231 | 61,323 | sky masks: **no clear gain** (3.4) |

Models/videos of the 5 finished jobs were saved to `data/CameraSet/results/` (git-ignored; models 12-30 MB,
videos 36-48 MB each). Download again at any time while the jobs exist:
`GET /jobs/<id>/model`, `GET /jobs/<id>/preview/download` (request the video with `POST /jobs/<id>/preview` first).

## 3. What was learned

### 3.1 COLMAP on video frames
* Matching was never the problem: for the dense8 set 714 of 715 consecutive frame pairs verified, median 4020
  inliers (`tools/server_diag/match_stats.py`). Pilot on 132 frames: median 1522 inliers between frames 0.5 s apart.
* The server undistorts and trains on the **largest** sub-model (not necessarily `sparse/0`); the others are
  thrown away. A video usually ends as one big model plus fragments - check with
  `tools/server_diag/inspect_job.sh <job_id>` how many images were really used (often only 30-75 %).
* **Do not judge a mapper run by its first sub-models.** The light job logged models of 13, 4, 23, 11, 30
  images and I predicted failure; it then produced a 231-image model. Look at the final result.

### 3.2 Speed: feature count dominates
* With the server's default 32k SIFT features per image the incremental mapper slowed to **~13 images/hour** at
  ~250 registered images of 1145 (each global bundle adjustment ran tens of minutes; projected > 3 days).
* The same data thinned to every 8th frame with `max_image_size 1280` (~12k features) finished the whole
  pipeline in **2 h** and still registered 231 images. Loop-detection indexing costs 3.5-4.7 s/image at 32k
  features (CPU) but ~1.5 s/image at 12k.
* Exhaustive matching of 572 images ran overnight; blocks of 50x50 images take ~52 s each at 32k features.

### 3.3 Server behaviour that bit
* `/data` (492 GB) was **100 % full** (171 job dirs; 58 % of the space was intermediate `model_<step>.ply`
  exports). COLMAP then dies with `SQLite error: database or disk is full`. Check `df -h /data` before big jobs.
* One sequential-matcher run died with `SQLite error: database disk image is malformed` although the disk was
  healthy and `PRAGMA quick_check` returned `ok`; a plain `POST /jobs/{id}/restart` (fresh database) fixed it.
* `POST /jobs/{id}/cancel` does not stop COLMAP/Brush **and** the single worker stays blocked until that process
  exits, so nothing else can start. Stopping it means `sudo systemctl restart gs-server` - but on start the
  service **re-queues every job found in `colmap_running`/`training`**, so cancel the job first (status
  `cancelled` is not re-queued), then restart. Only do this when no other job is active.
* `POST /jobs/{id}/preview` for a `pointcloud_only` job needs `transforms.json` in the dataset dir; for a
  COLMAP-format dataset create it with `tools/sky_mask/colmap_to_transforms.py`.
* On the VM `/home/anoma/gs_server` is a folder that itself has an `__init__.py`; the real package is
  `/home/anoma/gs_server/gs_server`. Scripts importing `gs_server` must put `/home/anoma/gs_server` first on
  `sys.path`. The service runs `python3.11`; the system `python3` has no OpenCV.
* Job data lives in `/data/jobs` (`GS_JOBS_DIR`), not under `/home/anoma`.

### 3.4 Visual quality and the sky-mask experiment
* Good: sidewalk slabs with tree shadows, hedges, bushes, parked cars, tree trunks.
* Weak (all models): black sky and far field (nothing there to render - **masks do not change this, the
  background colour is set by the renderer**), bright vertical streaks (a lamp pole) and rays at the edge of the
  reconstructed area, slightly smeared foreground sidewalk.
* Classical sky masks (`tools/sky_mask/sky_mask.py`: blue/bright/smooth region connected to the image top):
  correct on open sky above houses, misses sky seen through branches, no false positives on white walls. Only
  ~25 % of the frames contain much sky (mean 5.4 %, 14 of 231 images > 20 %). Effect on the model: splat count
  unchanged (61,320 -> 61,323), a crude "bright pixels in black area" proxy -24 %, but the lamp-pole streaks
  stayed and the two flythroughs are not pixel-aligned, so treat as **no measurable improvement**.

## 4. Server housekeeping done in this session

* Deleted through the API (all irreversible): 2 stale app jobs from 2026-09-11 (`5ec53883`, `954da524`);
  `b6706a9c` (failed), `d060fcbf` (failed), `c1eaf30d` (AU_Las_Vegas_anchor_smoketest2), `ea4dcf8e`
  (AU_Las_Vegas_aniso_reg_test); then 63 more: 59 failed/cancelled/stuck jobs and 4 older runs
  (`665ddd60` **AU_Las_Vegas - the baseline of the AU investigation**, `02548dc5`, `45ec5d06`, `657a4cae`).
  The AU report's comparison images are saved in `report_images/`, but the baseline model itself is gone.
* My own temporary jobs (the failed dense5 attempts) were deleted too.
* State at the end: **106 jobs** (105 completed, 1 `training_failed`), no active job, `/data` ~88 % used
  (~60 GB free at the last check on 2026-10-15; `gcloud` auth had expired afterwards, re-check with
  `df -h /data`).
* Left on the VM (small, safe to delete after a look): `/data/datasets/CameraSet_sky8` (dataset of the
  sky-mask job, 337 MB + transforms.json - **the `fc7ec144` job's `dataset_path` points here**, keep it while that
  job's preview/camera_positions are needed), `/home/anoma/cs_*.py|.sh` and `/home/anoma/cs_sky/` (helper
  scripts, now superseded by `tools/`), `/tmp/cs_comp_*`.

### 4.1 27 jobs deliberately kept (102 GB) - decide before deleting

They are referenced by committed documents/code, so deleting them would orphan those references. The owner
chose to keep them for now.

| ID | Job name | GB | Referenced by |
|---|---|---|---|
| `210ca694` | core_propellers_2_hybrid_known_poses_test6_adaptive | 8.8 | `gs_server/config.py`, `floater_filter_v2_test.py` (reference model) |
| `5afde525` | Core_Artifact_Left_colmap | 8.7 | Core_Artifact reports |
| `e140b494` | Core_Artifact_Right_nocolmap | 5.2 | Core_Artifact reports |
| `2151e56b` `6321bba9` `9a181af7` `56bf62dd` `d90ce883` `54de76d2` `bee9c851` `02056964` `54520125` `07c03e9b` `be2d94d8` `dd055e8c` | the 12 `*_nocolmap` jobs of 2026-08-11 (core_machines_next_to_long_tube, core_propellers_2, TY_UpperDeckAFT_UseThisOne, SSUS_AFT_Engine_Room_ES22/P2, C_Deck_FWD, D_Deck_FWD_Section_1/2, front_tip_full, inside_big_towers, long_silver_tube, engine_room_b_deck) | 5.2-5.3 each | Batch12 report |
| `79083d73` `645c1289` `a44e3232` `7e85701c` `ce743f9a` `9af738a3` `b475bfdf` `d9c6ff65` `df7303de` `6c0097bd` `8c872a47` | the matching 11 `*_colmap` jobs of 2026-08-11 | 0.5-2.7 each | Batch12 report |
| `1c9736d1` | TY_UpperDeckAFT_UseThisOne_colmap (training_failed) | 0.2 | Batch12 report (failure example) |

Cheapest way to get space back without deleting a single job: remove the intermediate `model_<step>.ply`
exports (257 GB on 2026-10-13, see `tools/server_diag/job_sizes.py`). The API cannot do that; it needs a shell
on the server and the owner's go-ahead. Delete only where a `model_<final>_filtered.ply` (or the final raw
model) exists - 55 old jobs have no `_filtered` file because the filters did not exist yet.

## 5. Reproduce / continue

```bash
# 1. frames: sharpest frame of every 8th window, one prefix per video
python tools/client/video_to_frames.py --out frames8 --block 8 \
    --video "data/CameraSet/<Scan 01>.MOV=scan01" --video "data/CameraSet/<Scan 02>.MOV=scan02"
# 2. job (light COLMAP settings that worked)
export GS_SERVER=http://<EXTERNAL_IP>:8080
python tools/client/submit_job.py --name CameraSet_light --mode colmap_images --frames frames8 \
  --colmap-config '{"single_camera": true, "matcher_type": "sequential", "sequential_overlap": 20,
                    "sequential_loop_detection": true, "max_image_size": 1280}'
# 3. wait, render the flythrough, download video + model
python tools/client/wait_for_job.py <JOB_ID> --preview --out results --label CameraSet_light
```

## 6. Ideas not done yet

1. Pruning splats far from the capture trajectory / a radius filter to remove the far-field streaks.
2. A coloured background (sky gradient) in `preview_renderer.py` instead of black.
3. A held-out evaluation split (every 8th training photo excluded) - still the missing prerequisite for any
   honest quality comparison.
4. Ask for Dot3D/AuraGO exports with posed images, then train with `hybrid_known_poses` + the LiDAR clouds.
5. Remove the intermediate exports on the server (see 4.1).
