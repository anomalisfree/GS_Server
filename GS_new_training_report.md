# Gaussian Splatting Training Report

## Resources

| | Link |
|---|---|
| Source data + COLMAP | [GS\_new — Google Drive](https://drive.google.com/drive/folders/1ZLy9r2aXifwVOp4-lA_FuYMxMbhP4_ry?usp=sharing) |
| Training outputs (.ply) | [GS\_new\_exports — Google Drive](https://drive.google.com/drive/folders/1HU7nZk4mx5f_OTeG_ScT6mjXfpV5ChDH?usp=sharing) |
| Interactive viewer | [New test scene — SuperSplat](https://superspl.at/scene/c0281e14) |

---

## Scene

**Interior of a USPS post office branch** — a medium-sized lobby featuring:
- A wall of PO boxes (mailboxes)
- An emergency exit door (marker #350)
- Notice boards ("Ship Smart. Ship Shape.")
- A USPS self-service kiosk
- The characteristic red horizontal stripe on the walls
- A drop ceiling with uniform fluorescent lighting
- ArUco markers on the floor (#335, #350) — ground control points for spatial registration

---

## Input Data

| Parameter | Value |
|---|---|
| Frames | **304 JPEGs** (frame_000000 – frame_000303) |
| Source | Mobile phone video, portrait orientation, handheld |
| Camera model | `SIMPLE_RADIAL` |
| Max SIFT features | 8192 per frame |
| Rig files | `rigs.bin`, `frames.bin` — indicate a multi-camera rig setup |

**Critical capture limitation:** all frames were taken along a **single linear trajectory** — the operator walked through the room once with no side passes, no circular sweeps around objects, and no upward-facing shots. This results in:
- Most surfaces (side walls, tops of mailboxes, ceiling) visible only at grazing angles or not at all
- COLMAP unable to reliably triangulate points seen from fewer than 2–3 distinct positions
- The GS model having no training signal for viewpoints not covered during capture

---

## COLMAP Stage — several hours

**Why it took so long:**
- GPU-accelerated SIFT with affine shape estimation (`estimate_affine_shape=true`) — slower but higher-quality descriptors
- Sequential matching + **loop detection** with a 256K vocabulary tree — required to correctly close the trajectory loop
- Bundle Adjustment across all 304 cameras with a GPU solver
- Full MVS pipeline → `fused.ply` (dense point cloud) + `meshed-poisson.ply` (Poisson mesh)

**Outputs:**
- `images.bin` — **67 MB** (poses for all 304 cameras)
- `points3D.bin` — **22 MB** (sparse 3D point cloud)
- `database.db` — **528 MB** (all SIFT features and matches)

---

## Gaussian Splatting Training (Brush) — ~20 minutes

**Parameters:** `--max-resolution 1024` + `--subsample-points 4` — reduced due to VRAM constraints on the RTX 2070 Super (8 GB); at full resolution the GPU was at 95.7% capacity (7849/8192 MB), causing the training to hang.

The contrast with COLMAP is striking: **30,000 Brush iterations completed in roughly 20 minutes**, versus several hours of preprocessing. This is typical for GS pipelines — the bottleneck is geometry reconstruction (SfM/MVS), while the actual neural representation training on GPU is fast.

**Splat count growth** (inferred from PLY file sizes):

| Iteration | File size | Growth |
|---|---|---|
| 5,000 | 344.9 MB | — |
| 10,000 | 433.3 MB | +88.4 MB |
| 15,000 | 487.8 MB | +54.5 MB |
| 20,000 | 487.8 MB | **0** |
| 25,000 | 487.8 MB | **0** |
| 30,000 | 487.8 MB | **0** |

Splat growth stopped entirely at iteration 15,000 — when densification is disabled by default (`growth_stop_iter=15000`). The final size of ~487.8 MB corresponds to approximately **8–8.5 million Gaussians**. From iteration 15k to 30k, only parameter optimization of existing splats took place.

**SuperSplat upload:** the file weighs **67.47 MB** — a compressed `.compressed.ply`.

---

## Potential of External LiDAR Point Cloud Initialization

COLMAP builds its sparse point cloud photogrammetrically — by matching visual features across frames. An external LiDAR scanner (e.g. Leica RTC360 or BLK360 — suggested by the presence of `rigs.bin` in the data) produces a **dense, metrically accurate point cloud** with no image processing required.

Using a LiDAR cloud as initialization instead of COLMAP:
- **Eliminates the SfM/MVS stage entirely** — hours become minutes or seconds
- Provides a **far denser initial point cloud** → densification saturates faster, requiring fewer training iterations
- Delivers **correct metric scale and orientation** from the very first iteration
- Covers areas **invisible from the camera trajectory** — ceiling, around corners, narrow passages

Overall, a LiDAR → GS pipeline is potentially **several times faster** than a full COLMAP pipeline, with better geometric quality. Camera poses are still required (from COLMAP or from the LiDAR scanner's own registration), but the heavy MVS stage is eliminated entirely.

---

## Result Quality & Recommendations for Recapture

**What worked well:**
- Uniform diffuse lighting — minimal specular highlights, GS reproduces this well
- Textured surfaces (notice boards, signage, mailboxes) — good detail
- Training completed in full (all 30k iterations, 6 checkpoints saved)

**Where artifacts are expected:**
- Areas not visible from the capture trajectory — floating splats or voids
- Side walls and ceiling — seen only at grazing angles → blurry, lacking clear geometry
- Quality degrades sharply when viewing from angles not present in training data

**Recommendations for recapture:**
1. Multiple parallel passes — zigzag across the full width of the room
2. Dedicated passes along each wall, camera facing perpendicular to the surface
3. Circular sweeps around key objects (kiosk, mailbox wall)
4. At least several frames with the camera tilted upward to capture the ceiling
5. Landscape (horizontal) orientation — generally preferable to portrait for interior scenes
