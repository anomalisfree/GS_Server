# CLAUDE.md

Operational notes for working on this repo as an agent. `README.md` covers
setup/API basics for a human developer; this file covers what you need to
actually operate the **live production server** and pick up where past work
left off. If something here goes stale, fix it in the same commit that makes
it stale.

## Production deployment

- VM: `gs-training-server`, zone `us-central1-a`, GCP project `dgs-training`.
  SSH via `gcloud compute ssh gs-training-server --zone=us-central1-a --project=dgs-training --quiet --command="..."`.
- The server code runs from `/home/anoma/gs_server/gs_server/` (note: nested
  one level deeper than the repo root name suggests) as user `anoma`,
  managed by systemd unit `gs-server.service`. `systemctl cat gs-server`
  shows the full unit including baked-in env vars (CUDA paths,
  `TORCH_CUDA_ARCH_LIST=8.9`, etc.)
- **Deploying a code change**: `gcloud compute scp <local file>
  gs-training-server:/home/anoma/gs_server/gs_server/<file>.py ...` then
  `sudo systemctl restart gs-server`. The service does **not** hot-reload -
  editing a `.py` file on disk does nothing until restarted.
- **Before restarting the service, always check for an active training job**
  (`GET /jobs`, filter for `status` not in `completed/failed/cancelled`) - a
  restart kills whatever COLMAP/Brush subprocess is running mid-job with no
  resume. There's a `check_jobs.py`-style helper pattern used throughout this
  session's transcript for this.
- `gcloud` auth expires every so often (session-dependent, not time-fixed) -
  if a command fails with "Reauthentication failed: cannot prompt during
  non-interactive execution", ask the user to note the browser opened, run
  `gcloud auth login alex.lisavets@q3dsensing.com` (the browser flow needs the
  user physically present, signed into **that** account). The machine also has
  `anomalisfree@gmail.com` credentialed, which lacks `compute.instances.get`
  on this project - if that one is active you get a permission error, not an
  auth error (`gcloud config set account alex.lisavets@q3dsensing.com`). Start
  only one login at a time: a second overlapping attempt corrupts the OAuth
  state ("CSRF Warning! State not equal").
- **Job API request shape.** `POST /jobs` takes `colmap_config`, `brush_config`
  and `hybrid_config` as **top-level** fields. A nested `{"config": {"brush":
  ...}}` is silently dropped by Pydantic (the job then trains with defaults
  and nothing errors) - check `GET /jobs/{id}` shows your options under
  `config` before waiting an hour. `hybrid_config` is the
  `hybrid_known_poses` step's options (`sor_k`, `overexposure_mask_enabled`,
  `person_mask_enabled`, `person_mask_dilate_px`, ...).
- `POST /jobs/{id}/cancel` marks the job cancelled but does **not** kill the
  COLMAP/Brush subprocess - it keeps running orphaned. Kill it by PID
  yourself before restarting the service or launching the next job.
- Server-only settings live in the systemd unit's `Environment=` lines, e.g.
  `COLMAP_VOCAB_TREE_PATH=/home/anoma/colmap_vocab/vocab_tree_flickr100K_words32K.bin`
  (COLMAP's official 32K vocab tree). `sequential_loop_detection` is per-job,
  but the vocab tree **path** is only read from this global setting - putting
  `vocab_tree_path` in a job's `colmap_config` is silently ignored. Edit the
  unit with `sudo`, then `systemctl daemon-reload` and restart.

## Gotcha: Git-Bash / MSYS path mangling in `--command=`

If you're driving this from a Bash tool on Windows (Git-Bash/MSYS), any
`--command="..."` string whose **first character is `/`** gets silently
mangled - MSYS treats a leading `/xxx` argument as a POSIX path and rewrites
it to a Windows path before it ever reaches `gcloud`/`plink`, so the remote
shell receives garbage like `C:/Program Files/...` and fails with `bash:
line 1: C:/Program: No such file or directory`. This is **not** an SSH
connectivity problem despite what gcloud's error message suggests.

Fix: never start the command string with an absolute path. Use the bare
command name (rely on `PATH`) instead of `/usr/local/bin/colmap ...` -
`colmap ...` works fine. An absolute path appearing *later* in the string
(e.g. as an argument) is fine; it's only the leading character that triggers
this.

A different, unrelated exit-code trap: a command whose last pipeline stage
is `grep` (e.g. `ps aux | grep foo | grep -v grep`) exits 1 when nothing
matches, which gcloud also reports as an "SSH connectivity" error. Read the
actual output before assuming the connection dropped.

## Gotchas: PowerShell quoting, broken Bash PATH, truncated argv

- In PowerShell, a `--command="..."` containing `$VAR`, nested quotes, `for`
  loops or `python -c '...'` gets mangled before it reaches the server (`$HOME`
  expands to the Windows home, quotes vanish). Don't fight it: write the
  remote logic to a local `.sh`/`.py` file, `gcloud compute scp` it to the
  server, and run it with a short `--command="python3.11 /home/anoma/x.py"`.
  Same for JSON request bodies: write a `.json`, scp it, `curl --data @file`.
- Some sessions start with a Bash tool whose `PATH` lacks `/usr/bin` (even
  `ls`/`tail`/`bash` are "not found"). Use the PowerShell tool instead.
- `ps aux` and journald truncate long command lines at the same spot, which
  looks exactly like "my flag wasn't passed". To see the real argv of a
  running process use `cat /proc/<pid>/cmdline | tr '\0' ' '`.
- Running GPU/gsplat scripts by hand over SSH needs the unit's env, or the
  gsplat JIT fails with "Ninja is required": `export
  PATH=/usr/local/cuda-12.1/bin:/home/anoma/.local/bin:$PATH; export
  TORCH_CUDA_ARCH_LIST=8.9` first.

## The Brush (Rust) changes are NOT in this repo

`brush/` is a separate repository (gitignored here). Everything below that
touches Rust - the depth render channel, position-anchor / aniso-reg /
photo-consistency in `train.rs`, the optional PLY properties in `export.rs`,
the new CLI flags in `config.rs` - exists **only as uncommitted edits in the
server's `/home/anoma/brush` working tree** (base commit `3edecbb`, 19 modified
files across `brush-dataset`, `brush-process`, `brush-render`,
`brush-render-bwd`, `brush-serde`, `brush-train`, `brush-ui`). Your local
`brush/` checkout does not have them. If that disk is lost the work is lost:
`git diff` on the server (and commit it to a Brush branch) before doing
anything risky. Build with `source /home/anoma/.cargo/env && cargo build
--release --bin brush_app` (~3.5 min); `GS_BRUSH_EXE` already points at
`target/release/brush_app`, so no redeploy step after a build.

## Major subsystems added this session

- **`gs_server/preview_renderer.py`** (new file) - on-demand video preview
  rendering (`POST /jobs/{id}/preview`, strictly opt-in, never automatic) and
  a cheap synchronous camera-pose endpoint (`GET /jobs/{id}/camera_positions`).
  Renders with `gsplat` (real tile-based CUDA rasterizer) when available,
  falling back to a hand-rolled EWA renderer otherwise. Read the module
  docstring before touching camera-pose math - the COLMAP-vs-original-frame
  distinction between training modes is easy to get backwards (it has been,
  twice, before this file existed).
  - `gsplat` needed a real CUDA 12.1 toolkit (`/usr/local/cuda-12.1`, already
    installed on the VM) on `PATH` and `TORCH_CUDA_ARCH_LIST=8.9` to
    JIT-compile against the L4 GPU - the system default `nvcc` (11.5) can
    neither target Ada (compute_89) nor parse GCC 11 headers. Both are baked
    into the systemd unit's `Environment=` now; if `gsplat` ever needs a
    fresh JIT rebuild (e.g. after clearing `~/.cache/torch_extensions`), it
    must happen with those same env vars in scope or it breaks the same way
    it did before.
  - Passing raw SH coefficients to `gsplat`'s own `sh_degree=` evaluation
    caused visible full-frame color-flicker on floater-heavy models (it
    doesn't clamp output to `[0,1]`, unlike `_sh_to_color` in this file,
    and `gsplat`'s real unnormalized alpha compositing lets an overshoot
    dominate a frame). Colors are computed per-frame via `_sh_to_color`
    instead - see `render_gsplat_frame`.
- **`gs_server/floater_filter.py`** - two independent, chained filter passes
  (see its module docstring): `filter_needle_floaters` (shape-based, existing)
  and `filter_isolated_clusters` (new - spatial-connectivity-based, catches
  scattered debris around the scan periphery that shape-based filtering can't
  see). The connectivity graph is built from a **bounded k-NN query**, not
  `cKDTree.query_pairs` with a radius - the latter blew up memory (OOM-killed
  the process silently, with no Python traceback) on a real 2.37M-splat dense
  model. Don't revert that without re-testing on a large model.
- **Path smoothing** (`preview_renderer._smooth_waypoint_path`) - the video
  flythrough's camera path is now smoothed with a clearance-aware moving
  average (won't smooth a waypoint through nearby scene geometry). Tunable
  per-request via `PreviewRequest.path_smoothing`.
- **Opt-in training options** (all default off, forwarded by `brush_runner.py`
  only when set; defined in `config.py` `BrushConfig` + Brush's `config.rs`):
  `depth_loss_weight`/`depth_max_range` (LiDAR depth supervision; the range cap
  matters - uncapped far readings blacked out a domed ceiling),
  `position_anchor_weight`/`position_anchor_max_dist`, `aniso_reg_weight`/
  `aniso_reg_ratio_threshold`, `photo_consistency_enabled`/
  `photo_consistency_warmup_steps`. **Normalization lesson (hit three times):**
  any penalty summed with `.sum()` over ~1.7M splats must be normalized by the
  count of *violating* splats, not all visible ones - otherwise it either
  explodes (NaN, every splat pruned within ~200 steps) or is diluted to
  nothing with no error. Follow the `violator_weight` pattern in `train.rs`.
  None of these moved the visual quality on AU_Las_Vegas; they are kept
  because they now run correctly, not because they help.
- **`photo_consistency_enabled`** accumulates, per splat, the render-vs-photo
  residual at the splat's own projected pixel from the warmup step on, and
  exports it as the optional PLY properties `photo_resid_avg`/
  `photo_resid_count` (absent when off: output is byte-identical). Its
  consumer `floater_filter.filter_photometric_inconsistency` is **off by
  default** (`photo_consistency_filter_enabled=False`): measured against the
  real photos it *lowers* PSNR (18.15 -> 17.53 dB at percentile 97.5), because
  the top-residual splats are mostly large blurry foreground/haze splats that
  still carry real image content.
- **Person masks** (`hybrid_config.person_mask_enabled`, default off,
  `known_pose_triangulation._generate_person_masks`): DeepLabV3 person class
  at 1024 px, upsampled to full resolution, dilated (`person_mask_dilate_px`,
  default 25), AND-ed into `hybrid_dir/masks/<stem>.png` (Brush's hard
  per-pixel loss mask: 0 = excluded, 255 = kept) on top of the overexposure
  masks. Costs ~3 min. The only change that visibly improved AU_Las_Vegas
  (ghost of a walking person gone, translucent veil weaker, +0.24 dB on
  static pixels). Note `mask_generator.py`'s masks are COLMAP-only and never
  reach Brush's loss.
- **`known_pose_triangulation.py` resolves each frame's image as
  `dataset_dir/images/<basename of file_path>`** - one flat directory, no
  subfolders. Images placed in e.g. `synthetic/` produce dangling symlinks and
  a COLMAP `point_triangulator` crash ("img_N.jpg does not exist in
  database"). To add images, make `images/` a real directory of symlinks plus
  the new files.
- Full protocol/examples for the preview/camera-position endpoints and path
  smoothing (not for the training options above - those are only documented
  here and in `config.py`): `CLIENT_TRAINING_MODES_DEVELOPER_GUIDE.md`,
  sections 12-13. Keep that file
  in sync with the actual `PreviewRequest`/response schemas in `models.py` -
  it has gone stale at least once already (defaults drifted after tuning).

## Known open issues (not yet solved)

- **Validate a suspected artifact against a real photo before spending time
  on it.** A long AU_Las_Vegas investigation chased "floating floor tiles"
  by the fountain using only renders; the white diamonds with black borders
  turned out to be real floor inlay that the model reproduces tile for tile.
  Method: pick the real training photo whose camera is nearest the render
  pose (position + look direction), render the model from that exact camera,
  compare side by side. Eight mechanisms (shape/position/connectivity filters,
  position-anchor, aniso-reg, depth loss, synthetic point-cloud renders fed to
  COLMAP twice, photo-consistency filter) were built and tested against that
  non-problem; the engineering fixes remain valid, the conclusions about "the
  artifact" do not.
- What is actually visibly wrong in AU_Las_Vegas renders: a semi-transparent
  haze/veil in mid-air and ghosts where people walked through the capture
  (person masks help, see above), and overall fidelity of only ~19 dB PSNR on
  *training* views (unmasked pixels), worst near the fountain (frames ~373-540:
  glossy moving water, mirror-like marble). Camera poses are accurate to well
  under a pixel (render-vs-photo phase-correlation shift median 0.4 px) and
  per-photo exposure differences are not a significant cause (a two-pass
  exposure normalization gave +0.014 dB) - don't re-investigate either.
- **There is no held-out evaluation split**: every PSNR above is on training
  views, so under- vs over-fitting can't be told apart and hyperparameter
  tuning can't be judged honestly. Excluding e.g. every 8th photo from
  training is the missing prerequisite.
- Don't expect point-cloud renders to substitute for photos in COLMAP: two
  full-SfM runs on 703 real + 432 point-cloud-rendered images (exhaustive
  matching, then sequential + overlap 50 + vocab-tree loop detection) fragmented
  into 8 and 12 disconnected sub-models, and every synthetic image that
  registered at all landed in a cluster with zero real photos. The renders are
  too photometrically unlike real photos for SIFT to bridge them.
- Full-SfM (`colmap_images`) on this venue fragments even with real photos
  alone (best sub-model 509-614 of 703 images); `hybrid_known_poses` exists
  precisely to avoid that and should stay the mode for AU_Las_Vegas.
- Some models are locally too transparent in places that shouldn't be, even
  after the filter passes - confirmed **not** caused by filtering (opacity
  distribution is statistically identical between kept/removed splats). See
  `config.py`'s `opac_decay`/`floater_filter_*` comments for what's been tried.
- Thin "needle" fringe artifacts along object silhouette edges were reported
  to survive the filters - believed photometric (blended edge pixel) rather
  than geometric. Unverified against real photos (see the first bullet).

## Cleanup convention

`scratchpad/` (gitignored) is scratch space for one-off debug scripts, test
renders, and downloaded comparison assets - never commit anything from it,
and feel free to clear it out entirely between sessions (nothing in it is
load-bearing). Loose one-off log/tracking files at the repo root
(`*_log.txt`, `*_created.txt`, `*_tracking.txt`) from past manual batch
operations were removed in the same commit as this file - don't recreate
that pattern; if a batch script needs a durable record, put it in a report
markdown file instead of a loose tracked `.txt`.
