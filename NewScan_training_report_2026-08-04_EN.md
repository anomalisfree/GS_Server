# NewScan Training Report (without COLMAP)

Date: 2026-08-04  
Project: GS Server / Brush  
Job ID: 6184365f-bf47-493a-8889-1b3172985e7a

## 1. Executive Summary

Training completed successfully in no-COLMAP mode (using precomputed poses + an existing point cloud).  
Final status: completed.  
Final export: model_30000.ply.  
Final splat count: 2,086,420.

## 2. What Was Trained and What Was Used (No COLMAP)

A direct Nerfstudio-format dataset was used on the server at:
- /data/uploads/NewScan

Dataset contents:
- images/*.jpg (frames)
- transforms.json (camera poses and intrinsics)
- points3d.ply (input cloud for GS initialization)

Pipeline:
- skip_colmap=true
- dataset_path=/data/uploads/NewScan
- direct launch in Brush

Important: COLMAP was not run for reconstruction in this training.

## 3. What Was Done and Why

### 3.1 Issue 1: Server freezing/lagging at startup

Cause:
- The original point cloud was too large for practical startup (276,389,920 points)
- Very long initialization reduced API responsiveness

Fix:
- Stepwise reduction of input points3d

Downsampling steps:
- original: 276,389,920 points
- intermediate: 7,896,855 points (35.00x smaller, -97.14%)
- final working version: 1,988,417 points (139.00x smaller, -99.28%)

Why downsampling was needed:
- to speed up initialization
- to reduce risk of server hangs
- to keep the run within practical L4 resource limits

### 3.2 Issue 2: training_failed with chunk longer than limit

Error:
- Training error: Separator is found, but chunk is longer than limit

Cause:
- Long subprocess stdout lines (Brush) exceeded asyncio read limits

Fix:
- Reduced Rust log verbosity for child process
- Increased subprocess stdout read buffer limit in the runner

Effect:
- chunk-limit error eliminated
- run reached completion

## 4. Training Parameters

Job parameters:
- total_steps: 30000
- max_resolution: 1024
- eval_every: 500
- export_every: 1000
- refine_every: 100

Actual result:
- current_step: 30000/30000
- splat_count: 2,086,420
- model_size_bytes: 492,396,672

## 5. Runtime

From API:
- started_at: 2026-08-04T13:14:47.845152
- completed_at: 2026-08-04T13:45:38.519240
- elapsed_seconds: 1850.7 (about 30 min 51 sec)

### 5.1 At Which Iteration Visible Changes Mostly Stopped

Based on indirect export metrics (PLY size stabilization) and the `stop_growth_at=12000` setting:
- active growth/model changes were observed up to about iteration 12,000
- from 12,000 to 30,000 export size remained stable (492,396,672 bytes)

Timing to saturation point:
- timestamp at 12,000: 2026-08-04T13:27:59.099576
- start -> 12,000 elapsed: about 13 min 11 sec

After saturation point:
- segment 12,000 -> 30,000 took about 17 min 39 sec
- this segment was mostly fine-tuning/stabilization, without noticeable model size growth

## 6. Resources Used

Infrastructure:
- VM: gs-training-server (GCP)
- External IP: 136.64.198.146
- Internal IP: 10.128.0.3

GPU:
- NVIDIA L4
- Driver: 535.309.01
- VRAM total: 23,034 MiB

CPU/process:
- training ran in brush_app under the gs-server systemd service

## 7. Where to Download Models

Base API:
- http://136.64.198.146:8080

Final model:
- http://136.64.198.146:8080/jobs/6184365f-bf47-493a-8889-1b3172985e7a/model

Final model info:
- http://136.64.198.146:8080/jobs/6184365f-bf47-493a-8889-1b3172985e7a/model/info

List of all intermediate exports:
- http://136.64.198.146:8080/jobs/6184365f-bf47-493a-8889-1b3172985e7a/exports

Download a specific iteration (template):
- http://136.64.198.146:8080/jobs/6184365f-bf47-493a-8889-1b3172985e7a/exports/model_XXXXX.ply

Intermediate models (available steps 01000..30000, step 1000):
- model_01000.ply
- model_02000.ply
- model_03000.ply
- model_04000.ply
- model_05000.ply
- model_06000.ply
- model_07000.ply
- model_08000.ply
- model_09000.ply
- model_10000.ply
- model_11000.ply
- model_12000.ply
- model_13000.ply
- model_14000.ply
- model_15000.ply
- model_16000.ply
- model_17000.ply
- model_18000.ply
- model_19000.ply
- model_20000.ply
- model_21000.ply
- model_22000.ply
- model_23000.ply
- model_24000.ply
- model_25000.ply
- model_26000.ply
- model_27000.ply
- model_28000.ply
- model_29000.ply
- model_30000.ply

Example direct link to an intermediate model:
- http://136.64.198.146:8080/jobs/6184365f-bf47-493a-8889-1b3172985e7a/exports/model_10000.ply

## 8. How to Improve Quality

### 8.1 Increase Image Resolution

Impact on quality:
- usually the strongest positive impact
- better texture detail and fine geometric features
- fewer artifacts when viewpoint coverage is sufficient

Expected effect:
- clear visual quality gain when moving from 320x240 to higher resolutions
- especially on object boundaries and fine textures

Cost:
- higher VRAM usage
- longer training time

### 8.2 Should the Number of Points Be Increased?

Short answer:
- yes, but not indefinitely
- after a certain threshold, quality improves slowly while instability and runtime increase significantly

Practical approach:
- keep a reasonable points3d seed size (not hundreds of millions)
- increase max_splats gradually (for example 2M -> 3M -> 4M -> 5M)
- compare quality/time/stability after each step

For the current dataset:
- upgrading frame resolution will usually help more than trying to load all original 276M points

### 8.3 What Else to Improve

- More diverse viewpoints with good overlap
- Stable exposure and no motion blur/shake
- Remove problematic frames (blurred/black/overexposed)
- Increase max_resolution in training if needed
- Tune growth parameters and max_splats if needed, while monitoring VRAM

### 8.4 Can Results Be Improved by Adding COLMAP While Keeping the Original Cloud

Short answer:
- yes, but only in a hybrid scenario with proper coordinate-system alignment
- you cannot simply enable COLMAP and provide an external cloud simultaneously without registration into the same coordinate frame

When COLMAP helps:
- when image-based reconstruction can produce more consistent poses and better overlap
- when RGB frame quality is sufficient (sharpness, texture, diverse viewpoints)

When gains are limited:
- if most frames are low-resolution (as in the current set dominated by 320x240)
- if the original cloud is extremely dense and overloads initialization

Recommended hybrid pipeline (COLMAP + external cloud):
- run COLMAP and obtain poses/sparse geometry
- estimate transform between external cloud and COLMAP frame: q = sRp + t
- transform external cloud into COLMAP frame
- downsample cloud to a practical size
- use it as points3d.ply for GS initialization
- train with controlled max_splats

Practical conclusion for NewScan:
- for the current dataset, the primary quality boost usually comes from improving RGB input quality (resolution/sharpness/coverage)
- COLMAP is useful as an additional source of consistent poses/geometry, but not a replacement for high-quality input frames
- using all original hundreds of millions of points is not practical: a moderate downsample plus gradual max_splats increase is preferable

## 9. Conclusion

A successful run was completed without COLMAP, using precomputed poses + an existing point cloud.  
The critical failure causes were eliminated:
- overly heavy points3d seed
- subprocess log read-limit overflow

After these fixes, the pipeline reliably reached the final model at step 30,000.
