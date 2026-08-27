# Core_Artifact (Left & Right) Training Report & Server Fixes

Date: 2026-08-07
Server: `http://136.64.198.146:8080` (GCP VM `gs-training-server`, project `dgs-training`)
Audience: client-side developers integrating with GS Server

## 1. Summary

Trained the `Core_Artifact_Left` and `Core_Artifact_Right` LiDAR captures, each two ways, as requested:

1. **No COLMAP** (`training_mode=pointcloud_only`) — using the scanner's own camera poses and point cloud.
2. **With COLMAP** (`training_mode=colmap_images`) — full SfM reconstruction from the images only.

All four runs completed successfully. While preparing the first run, found and fixed three real bugs in the
server/tooling — one of them (a 1000x unit mismatch between the point cloud and camera poses) silently
destroyed the very first training attempt even though the server reported `completed`. Details, fixes, and
final results below.

### 1.1 Note on `Left`'s image filenames

`Core_Artifact_Left`'s pose/intrinsics files (`*_09_38_35_*`) reference image filenames with a `09_38_35`
prefix, but the actual JPEGs on disk (and the ones in `Core_Artifact_Right/images`) are prefixed `09_11_56`
— no `09_38_35`-prefixed image ever exists on disk. Confirmed with the data owner this is expected: the
capture device stamps all photos with one continuous running timestamp, while `09_38_35` is a separate
export timestamp on the LiDAR registration pass. Frames are matched **by index**
(`SensorFrame_00000` ↔ `SensorFrame_00000`), not by filename prefix — this is not a data error.

## 2. Bugs Found & Fixed

### 2.1 500GB data disk was never mounted (infra)

`/data` (jobs, uploads, models) was living on the 100GB root disk, which was at 90% usage (10GB free) —
despite a dedicated 500GB disk (`/dev/nvme0n2`) being attached to the VM and provisioned for exactly this
purpose (per `deploy/create_vm.sh`). It had simply never been formatted/mounted.

**Fix:** formatted `/dev/nvme0n2` as ext4, migrated existing `/data` contents (44GB) onto it, added an
`/etc/fstab` entry, remounted. Root disk is now at 46% (54GB free), `/data` has 423GB free. No application
code changes — pure ops fix. Client impact: none directly, but this was very likely a contributor to prior
flakiness/hangs under load.

### 2.2 Point cloud was in millimeters, camera poses in meters — silent training collapse

This is the one that matters most for anyone using `pointcloud_only` or `hybrid_colmap_pointcloud` mode with
an externally-supplied point cloud.

The `Core_Artifact_Left_prepared` dataset (`transforms.json` + `points3d.ply`) had already been prepared on
the server from the raw scanner export before this session started. Its `points3d.ply` was **not**
downsampled (137.8M points, 2.5GB — the same order of magnitude that caused hangs in the earlier `NewScan`
job) and, worse, its coordinates were in a completely different unit/scale than the camera poses in
`transforms.json`:

| | X range | Y range | Z range |
|---|---|---|---|
| `points3d.ply` (raw) | -18675 .. 10121 | 715 .. 29804 | -2579 .. 13510 |
| camera poses (`transforms.json`) | -12.1 .. 7.1 | 0.5 .. 30.8 | -0.4 .. 0.5 |

Dividing the point cloud by 1000 (mm → m) lines up almost exactly with the camera-pose range (Y matches to
within 3%), confirming the point cloud was exported in millimeters while the SensorFrames poses are in
meters — a unit mismatch specific to whatever process generated this dataset's `points3d.ply` (the repo's
`leica_to_nerfstudio.py`, which sources points from E57 scans via `pye57`, doesn't have this problem since
E57 cartesian coordinates come out in meters already; this dataset's cloud came from a different,
CloudCompare-based export that didn't apply the conversion). `Core_Artifact_Right`'s point cloud did **not**
have this problem — it was already in meters, matching its own pose range directly.

**Consequence for Left:** Brush initialized ~2M Gaussians roughly 1000x farther from every camera than the
actual scene. The job **completed successfully from the API's point of view** (`status: completed`, no
error) but the model was garbage — during training, opacity/scale decay pruned 99.3% of the Gaussians (from
1,997,986 down to 13,404) by step ~8,400, then nothing happened for the remaining 21,600 steps. Nothing in
the job status surfaced this — `splat_count` was reported correctly, but there's no signal today that flags
"this final splat count looks abnormally low relative to the input cloud."

**Fix applied to the dataset:** rebuilt `points3d.ply` from the original raw scan, this time both
downsampled (137.8M → ~2M points) **and** correctly scaled (÷1000). Re-ran training — this time the splat
count stayed stable (see §3).

**Recommendation for client devs:** if your pipeline lets users supply their own `points3d.ply` (hybrid
mode, or a custom `dataset_path`), there is currently no server-side validation that the point cloud's
coordinate frame/scale matches the camera poses it's paired with. A garbage-but-"completed" job is a bad
failure mode. Consider either (a) documenting the meters convention clearly for anyone hand-preparing
datasets, or (b) a server-side sanity check (e.g. warn/fail if the point cloud's bounding box is >Nx larger
than the camera trajectory's bounding box) — not implemented yet, flagging as a suggestion.

### 2.3 `downsample_ply_memmap.py` assumed a fixed 6-field PLY layout

This repo utility (used to shrink oversized point clouds before upload — see the `NewScan` report's
276M→1.98M downsampling) hardcoded a `{x,y,z,red,green,blue}` dtype when memory-mapping the source file. The
`Core_Artifact_Left` cloud has an extra `scalar_Intensity` float property (common in CloudCompare exports),
which would have silently misaligned every record read past the header — corrupting, not just failing.

**Fix:** the script now parses the actual `property <type> <name>` lines from the PLY header into a dynamic
numpy dtype, and only emits `x,y,z,red,green,blue` to the output regardless of what extra properties the
source has. Also added a `--scale` flag (used for the mm→m fix above).

### 2.4 COLMAP `matcher_type` was not actually configurable per job

`gs_server/README.md` has always documented `colmap_config.matcher_type` as a valid per-job override (and
`CreateJobRequest.colmap_config` accepts arbitrary keys with no validation), but `colmap_runner.py` only ever
read `matcher_type` from the server's global `ColmapConfig` default (`"exhaustive"`), silently ignoring
whatever a client sent. There was also no env var to change the global default.

For large sequential captures (641/466 frames from continuous walks), exhaustive matching means ~200K+ image
pairs — impractically slow. Sequential matching (each frame matched against its temporal neighbors) is
effectively what you want for this kind of capture, and cut matching time down to a couple of minutes.

**Fix:**
- `colmap_runner.py` now reads `job.config["colmap"]["matcher_type"]` (falls back to server default).
- Added `sequential_loop_detection` (bool) — when true and a vocab-tree file is configured via the new
  `COLMAP_VOCAB_TREE_PATH` env var, enables COLMAP's loop-closure detection for the sequential matcher; if no
  vocab tree is configured it's silently skipped (logged) rather than erroring. (No vocab-tree file was
  installed on this server, so both COLMAP runs below used plain sequential matching without loop closure.)
- Added `COLMAP_MATCHER_TYPE` / `COLMAP_SEQUENTIAL_LOOP_DETECTION` / `COLMAP_VOCAB_TREE_PATH` env vars for
  server-wide defaults.
- Documented all of this in `gs_server/README.md` under "COLMAP Settings", including a table of when to use
  each matcher type.

**Client impact:** `colmap_config.matcher_type` and `colmap_config.sequential_loop_detection` are now real,
working per-job options. No breaking changes — anyone not setting them gets the previous behavior
(`exhaustive`).

### 2.5 Minor: `leica_to_nerfstudio.py` crashed on its final summary print on Windows

Used a Unicode `×` character that `cp1251`-codepage Windows consoles can't encode, crashing the script
*after* it had already finished writing `transforms.json` and copying images (i.e. cosmetic, output was
still correct, but it looks like a failure and breaks anything scripting around this tool). Fixed by using
ASCII `x` instead.

## 3. Results

All four jobs used `training_mode` as noted, `brush_config: {total_steps: 30000, max_resolution: 1024}`
(all other Brush params default — the anti-floater tuning from `gs_server/README.md`), and no masking.

| Run | Job ID | Mode | Images | Final splats | Elapsed |
|---|---|---|---|---|---|
| Left, no-COLMAP | `d4cd7c94-b155-4259-bc7f-f489b900eff9` | `pointcloud_only` | 641 (poses/cloud from scanner) | **2,001,835** | 30.0 min |
| Left, COLMAP | `5afde525-8048-44a4-93f5-f86973bc6b89` | `colmap_images`, sequential matcher | 641 | **890,907** | 17.0 min |
| Right, no-COLMAP | `e140b494-294b-4f79-8da6-86bacb890c7e` | `pointcloud_only` | 466 (poses/cloud from scanner) | **1,974,684** | 30.2 min |
| Right, COLMAP | `2d4e7366-798b-4b0d-af98-4283fab5e35c` | `colmap_images`, sequential matcher | 466 | **495,141** | 13.1 min |

(For reference: the first, broken Left attempt — job `f75e9403-baac-4546-9ee5-7fc9a20a3dec` — collapsed to
13,404 splats due to the mm/m bug in §2.2 and should be disregarded/deleted.)

**Splat-count pattern:** the no-COLMAP runs converge to noticeably more Gaussians (~2M, close to the input
point cloud size) than the COLMAP runs (~0.5-0.9M). This tracks with the anti-floater growth settings
(`growth_grad_threshold`, `growth_select_fraction`, `stop_growth_at=12000`) — Brush only grows Gaussians in
regions with strong position gradients, and a dense, pre-registered LiDAR cloud gives it far more seed points
in well-covered regions to grow from than COLMAP's sparser SfM point cloud does. This isn't automatically
"better" — more splats from a correct point cloud generally means finer detail and full coverage (including
areas the camera only grazed), while COLMAP's leaner reconstruction can still be geometrically accurate,
just less dense. Recommend a visual comparison in a viewer (e.g. SuperSplat) before picking a default mode
for production.

### Downloads

Base URL: `http://136.64.198.146:8080`

| Run | Model | Info |
|---|---|---|
| Left, no-COLMAP | `/jobs/d4cd7c94-b155-4259-bc7f-f489b900eff9/model` | `/jobs/d4cd7c94-b155-4259-bc7f-f489b900eff9/model/info` |
| Left, COLMAP | `/jobs/5afde525-8048-44a4-93f5-f86973bc6b89/model` | `/jobs/5afde525-8048-44a4-93f5-f86973bc6b89/model/info` |
| Right, no-COLMAP | `/jobs/e140b494-294b-4f79-8da6-86bacb890c7e/model` | `/jobs/e140b494-294b-4f79-8da6-86bacb890c7e/model/info` |
| Right, COLMAP | `/jobs/2d4e7366-798b-4b0d-af98-4283fab5e35c/model` | `/jobs/2d4e7366-798b-4b0d-af98-4283fab5e35c/model/info` |

Intermediate checkpoints (every 2,500 steps) are available at `/jobs/{job_id}/exports`.

## 4. Files Changed

- `gs_server/config.py` — added `matcher_type`/`sequential_loop_detection`/`vocab_tree_path` env overrides.
- `gs_server/colmap_runner.py` — wired per-job `matcher_type`/`sequential_loop_detection` through to the
  `*_matcher` COLMAP command.
- `gs_server/README.md` — documented the above, with a matcher-type selection table.
- `downsample_ply_memmap.py` — dynamic PLY property parsing (was hardcoded to xyz+rgb only) + new `--scale`
  flag for unit conversion.
- `leica_to_nerfstudio.py` — fixed a Windows console encoding crash in the final summary print.
- Server-side (ops, no code): `/data` migrated to the dedicated 500GB disk;
  `/data/uploads/Core_Artifact_Left_prepared/points3d.ply` replaced with a correctly-downsampled-and-scaled
  version; added `/data/uploads/Core_Artifact_Right_prepared` (images + `transforms.json` + downsampled
  `points3d.ply`).

All code changes are backward compatible — no existing client integration should need to change anything to
keep working as before; the fixes only make previously-broken/ignored options actually work correctly.

## 5. Open Recommendation

There's currently no automated check that a supplied point cloud's coordinate frame/scale is consistent with
the camera poses it's trained against (§2.2). This produced a job that reported `completed` with no error
while being structurally invalid. Worth adding a sanity check (e.g. compare bounding-box sizes/centers
between the point cloud and the camera trajectory, warn or fail if wildly mismatched) to
`job_manager.py`/`brush_runner.py` before training starts, so a bad dataset fails fast and visibly instead of
silently producing an unusable model.
