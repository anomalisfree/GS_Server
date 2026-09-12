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
  `gcloud auth login` (the browser flow needs the user physically present).

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
- Full protocol/examples for all of the above:
  `CLIENT_TRAINING_MODES_DEVELOPER_GUIDE.md`, sections 12-13. Keep that file
  in sync with the actual `PreviewRequest`/response schemas in `models.py` -
  it has gone stale at least once already (defaults drifted after tuning).

## Known open issues (not yet solved)

- Some models are locally too transparent in places that shouldn't be, even
  after both filter passes - confirmed **not** caused by filtering (opacity
  distribution is statistically identical between kept/removed splats). Most
  likely a training-time regularization or coverage issue, not a
  post-processing one. See `config.py`'s `opac_decay`/`floater_filter_*`
  comments for the history of what's already been tried here.
- Thin "needle" fringe artifacts along object silhouette edges survive both
  filters - believed to be a photometric (blended edge pixel) effect from
  training rather than a geometric outlier, per an earlier failed attempt at
  a distance-to-source-cloud filter (see git history / `config.py` comments).

## Cleanup convention

`scratchpad/` (gitignored) is scratch space for one-off debug scripts, test
renders, and downloaded comparison assets - never commit anything from it,
and feel free to clear it out entirely between sessions (nothing in it is
load-bearing). Loose one-off log/tracking files at the repo root
(`*_log.txt`, `*_created.txt`, `*_tracking.txt`) from past manual batch
operations were removed in the same commit as this file - don't recreate
that pattern; if a batch script needs a durable record, put it in a report
markdown file instead of a loose tracked `.txt`.
