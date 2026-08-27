# Core_Artifact (Left & Right): Compute Time, Iterations, Quality

Date: 2026-08-08
Server: `http://136.64.198.146:8080`

## 1. Summary Table

| Run | Job ID | Total (wall clock) | COLMAP | Brush Training | Iterations | Final Splats | Stabilized at Iteration |
|---|---|---|---|---|---|---|---|
| Left, no COLMAP | `d4cd7c94-...` | 30 min | — | 30 min | 30,000 | 2,001,835 | ~5,000 (17%) |
| Left, with COLMAP | `5afde525-...` | **10h 55min** | **10h 38min** | 17 min | 30,000 | 890,907 | ~11,800 (39%) |
| Right, no COLMAP | `e140b494-...` | 30 min | — | 30 min | 30,000 | 1,974,684 | ~4,800 (16%) |
| Right, with COLMAP | `2d4e7366-...` | **6h 19min** | **6h 06min** | 13 min | 30,000 | 495,141 | ~11,800 (39%) |

All 4 runs used identical parameters: `total_steps=30000`, `max_resolution=1024`, `refine_every=200`,
`export_every=2500`, `eval_every=1000`, default anti-floater settings (`growth_stop_iter=12000`, etc.).

## 2. Result Links

Base URL: `http://136.64.198.146:8080`

| Run | Final Model | Info | All Checkpoints |
|---|---|---|---|
| Left, no COLMAP | `/jobs/d4cd7c94-b155-4259-bc7f-f489b900eff9/model` | `/jobs/d4cd7c94-b155-4259-bc7f-f489b900eff9/model/info` | `/jobs/d4cd7c94-b155-4259-bc7f-f489b900eff9/exports` |
| Left, with COLMAP | `/jobs/5afde525-8048-44a4-93f5-f86973bc6b89/model` | `/jobs/5afde525-8048-44a4-93f5-f86973bc6b89/model/info` | `/jobs/5afde525-8048-44a4-93f5-f86973bc6b89/exports` |
| Right, no COLMAP | `/jobs/e140b494-294b-4f79-8da6-86bacb890c7e/model` | `/jobs/e140b494-294b-4f79-8da6-86bacb890c7e/model/info` | `/jobs/e140b494-294b-4f79-8da6-86bacb890c7e/exports` |
| Right, with COLMAP | `/jobs/2d4e7366-798b-4b0d-af98-4283fab5e35c/model` | `/jobs/2d4e7366-798b-4b0d-af98-4283fab5e35c/model/info` | `/jobs/2d4e7366-798b-4b0d-af98-4283fab5e35c/exports` |

Checkpoints are available every 2,500 iterations: `model_02500.ply` ... `model_30000.ply`.

## 3. Detailed COLMAP Stage Timing

### 3.1 Left (641 images)

| Stage | Start | End | Duration |
|---|---|---|---|
| Feature extraction | 15:18:43 | 15:19:29 | 46 sec |
| Feature matching (sequential) | 15:19:29 | 15:28:18 | 8 min 49 sec |
| **Sparse reconstruction (mapper)** | 15:28:18 | 01:55:07 (+1 day) | **10h 26min 49sec** |
| Image undistortion | 01:55:07 | 01:56:20 | 1 min 13 sec |
| Brush training | 01:56:20 | 02:13:21 | 17 min 01 sec |
| **Total** | 15:18:43 | 02:13:21 | **10h 54min 39sec** |

### 3.2 Right (466 images)

| Stage | Start | End | Duration |
|---|---|---|---|
| Feature extraction | 02:43:33 | 02:44:17 | 44 sec |
| Feature matching (sequential) | 02:44:17 | 02:49:33 | 5 min 16 sec |
| **Sparse reconstruction (mapper)** | 02:49:33 | 08:49:01 | **5h 59min 28sec** |
| Image undistortion | 08:49:01 | 08:49:48 | 47 sec |
| Brush training | 08:49:48 | 09:02:52 | 13 min 04 sec |
| **Total** | 02:43:33 | 09:02:52 | **6h 19min 19sec** |

### 3.3 Key Timing Finding

**The bottleneck is the incremental mapper (bundle adjustment), not matching.** Feature extraction +
matching (the stage sped up by our switch to the sequential matcher) took **under 10 minutes combined** in
both runs. The `mapper` stage itself accounts for **>95% of total COLMAP time** (10h27min of 10h38min for
Left, 5h59min of 6h06min for Right). COLMAP's incremental reconstruction recomputes a global bundle
adjustment as each new image is registered, which slows down sharply as the image count grows — hence the
near-linear relationship observed: 641 images → 10.5h, 466 images → 6h (not perfectly linear, but close).

Our exhaustive → sequential matcher change was correct and effective (see the earlier bug-fix report), but
it only speeds up ~2% of total COLMAP run time at this image count. To meaningfully speed up the COLMAP
mode, the only real lever is **reducing the number of images** (frame decimation) or finding a faster
mapper/bundle-adjustment configuration — optimizing matching further won't move the needle.

## 4. Iteration at Which the Result Stops Visibly Changing

Tracked the splat count at every Refine iteration (`refine_every=200`) up to the point where the count stops
growing and stays constant for the rest of training (30,000 steps). Exported checkpoint files
(`model_XXXXX.ply`) confirm the same — their file size also stops changing at the same point.

| Run | Starting Splats (iter ~200) | Final Splats | Stabilization Iteration | Growth During Growth Phase |
|---|---|---|---|---|
| Left, no COLMAP | 1,997,973 | 2,001,835 | **~5,000** | +0.19% |
| Left, with COLMAP | 858,507 | 890,907 | **~11,800** | +3.8% |
| Right, no COLMAP | 1,972,638 | 1,974,684 | **~4,800** | +0.10% |
| Right, with COLMAP | 486,050 | 495,141 | **~11,800** | +1.9% |

**Observation:** the no-COLMAP runs (initialized from the dense LiDAR point cloud) stabilize very early —
at **16-17%** of total training — and barely grow (< 0.2%), because the point cloud already covers the
scene densely and there's little left for densification to add. The COLMAP runs keep actively growing
splats **all the way to `growth_stop_iter=12000`** (39% of training) — meaning their growth is cut off by
the configuration limit, not by organic convergence. This suggests the COLMAP-based reconstruction could
potentially keep growing further if `growth_stop_iter` were raised — a signal that for COLMAP-based datasets
(sparser point cloud), it may be worth raising this parameter if maximum detail is needed, at the cost of
longer training.

After the stabilization point, all remaining iterations (up to 30,000) are pure refinement of the position,
color, and opacity of existing splats, with no change in their count.

## 5. Quality Assessment

**Observations:**

- **No-COLMAP runs retain 2.2-4x more splats** than COLMAP runs on the same data (2.0M vs 0.89M for Left;
  1.97M vs 0.50M for Right). This is a direct consequence of the source point cloud's density: the
  pre-registered LiDAR cloud gives densification far more seed points in well-covered regions. More splats
  isn't automatically "better," but it typically means more complete scene coverage (including areas poorly
  seen from the camera trajectory — grazing-angle walls, ceilings, etc.) and potentially finer detail where
  the cloud is accurate.
- **COLMAP runs are photometrically self-consistent by construction** — camera poses are obtained by
  minimizing reprojection error against the photos themselves, so the geometry is tightly tied to what's
  actually visible in the images. This is a real strength of the COLMAP mode, even with fewer splats.
- Both Left and Right show the same qualitative pattern (no-COLMAP denser and stabilizing early; COLMAP
  sparser but growing to the configured limit) — this is a systematic difference between the two modes under
  the current anti-floater settings, not a one-off artifact of a single dataset.

**Recommendation:** a final choice of mode (or a hybrid approach) should be confirmed with a visual review
in a viewer — the training-log data alone isn't sufficient for a quantitative "which is better" verdict. If
useful, I can configure an eval split in the datasets and rerun the final N iterations with PSNR/SSIM, or
prepare comparison renders if you specify a rendering tool/viewer for the `.ply` files.
