# tools/

Small, parameterised scripts that were written during the AU_Las_Vegas and CameraSet work and are worth
keeping. Everything that is *not* here lives in `scratchpad/` (git-ignored, throw-away).

Install once: `pip install -r tools/requirements.txt` (requests, opencv-python-headless, numpy, markdown).
Server address for the client scripts: `--server http://<IP>:8080` or `export GS_SERVER=...`
(IP: `gcloud compute instances describe gs-training-server ...`, see `CLAUDE.md`).

| Script | Runs on | Purpose |
|---|---|---|
| `client/video_to_frames.py` | your PC | video(s) -> JPEGs, the sharpest frame of every N-frame window; several videos with different prefixes can go into one job |
| `client/submit_job.py` | your PC | `POST /jobs` with correctly placed top-level config blocks, uploads images in batches, `--dry-run` to preview the request |
| `client/wait_for_job.py` | your PC | poll a job; `--preview` renders the flythrough; `--out` downloads video + model |
| `server_diag/colmap_models.py` | the VM | sizes of the COLMAP sub-models the mapper has produced so far (from the journal) |
| `server_diag/match_stats.py` | the VM | read-only quality check of a COLMAP `database.db` (verified pairs, inliers) |
| `server_diag/inspect_job.sh` | the VM | registered images / points / error per COLMAP model and how many images were trained on |
| `server_diag/job_sizes.py` | the VM | disk use per job, biggest first, incl. intermediate Brush exports |
| `sky_mask/` | the VM (python3.11) | **experimental**: copy a finished COLMAP set, add sky masks, make it trainable as `pointcloud_only`, add `transforms.json` for the preview. Little effect, see the CameraSet report |
| `docs/build_guide_html.py` | your PC | Markdown guide -> one self-contained HTML (images embedded, sidebar TOC), validates links |

Run the VM scripts like this (PowerShell mangles inline quotes - always copy a file, never inline code):

```bash
gcloud compute scp tools/server_diag/colmap_models.py gs-training-server:/home/anoma/ --zone=us-central1-a --project=dgs-training
gcloud compute ssh gs-training-server --zone=us-central1-a --project=dgs-training --quiet --command="sudo python3 /home/anoma/colmap_models.py '2026-10-13 15:44:00'"
```

Typical flow (details and numbers in `CameraSet_Training_Report_2026-10-19.md`):

```bash
python tools/client/video_to_frames.py --out frames8 --block 8 --video "A.MOV=scan01" --video "B.MOV=scan02"
python tools/client/submit_job.py --name my_scene --mode colmap_images --frames frames8 \
   --colmap-config '{"single_camera": true, "matcher_type": "sequential", "sequential_overlap": 20,
                     "sequential_loop_detection": true, "max_image_size": 1280}'
python tools/client/wait_for_job.py <JOB_ID> --preview --out results --label my_scene
```
