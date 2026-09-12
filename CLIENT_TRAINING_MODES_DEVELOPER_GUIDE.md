# GS Server: Developer Guide for Training Mode Selection

Date: 2026-08-05  
Audience: Client developers (web, desktop, mobile, Unity, backend integrators)

Current production API base URL: `http://136.64.198.146:8080`  
Note: IP/host may change between environments. Keep base URL configurable (env/config), not hardcoded.

## 1. Goal

This guide explains how to select one of three training modes from the client side:

1. colmap_images
2. pointcloud_only
3. hybrid_colmap_pointcloud

It also includes API payloads, validation rules, recommended UX flow, and ready-to-use examples.

## 2. Training Modes

### 2.1 colmap_images

Use case:
- You have only images.
- Server should run full COLMAP + Brush pipeline.

Input required:
- Uploaded images via /jobs/{job_id}/upload (+ /upload/complete).

Input not required:
- dataset_path
- pointcloud_path

Pipeline:
- EXIF normalization -> optional masking -> COLMAP -> Brush training.

### 2.2 pointcloud_only

Use case:
- You already have a prepared Nerfstudio/COLMAP-compatible dataset on the server.
- You want to skip COLMAP and train directly.

Input required:
- dataset_path (server filesystem path), for example: /data/uploads/NewScan

Input not required:
- uploaded images
- pointcloud upload endpoint

Pipeline:
- Skip EXIF/masking/COLMAP -> Brush training from dataset_path.

Important:
- dataset_path must already exist on the server and contain valid training data.
- If dataset_path is missing, job will fail with validation error.

### 2.3 hybrid_colmap_pointcloud

Use case:
- You want COLMAP camera solution from images.
- You also want to inject an external points3d.ply before Brush starts.

Input required:
- Uploaded images (/upload + /upload/complete).
- External point cloud path in pointcloud_path OR upload it via:
  - /jobs/{job_id}/pointcloud
  - /jobs/{job_id}/pointcloud/chunk

Pipeline:
- EXIF normalization -> optional masking -> COLMAP -> replace/inject dense/points3d.ply -> Brush training.

Important:
- If pointcloud_path is not provided for hybrid mode, job will fail.

## 3. API Changes Summary

### 3.1 New request fields

Create job payload now supports:
- training_mode: colmap_images | pointcloud_only | hybrid_colmap_pointcloud
- pointcloud_path: optional server path to external points3d.ply

Start job payload now supports:
- training_mode: optional override
- dataset_path: optional override
- pointcloud_path: optional override

Backward compatibility:
- Legacy skip_colmap=true still works and maps to pointcloud_only behavior.

### 3.2 New upload endpoints for hybrid mode

- POST /jobs/{job_id}/pointcloud
  - multipart/form-data
  - field name: file
  - supports .ply

- POST /jobs/{job_id}/pointcloud/chunk
  - multipart/form-data
  - fields:
    - filename
    - chunk_index
    - total_chunks
    - chunk

Server stores uploaded cloud as:
- <job_dir>/inputs/points3d.ply
and writes this value to job.config.pointcloud_path.

## 4. Recommended Client UX Flow

### 4.1 Step 1: Ask user for mode

Suggested mode picker labels:
- Images + COLMAP
- Existing Dataset (No COLMAP)
- Hybrid (COLMAP + External Point Cloud)

### 4.2 Step 2: Dynamic form fields

If mode = colmap_images:
- Require images folder/file selection.

If mode = pointcloud_only:
- Require dataset_path (server path).
- Hide image upload UI.

If mode = hybrid_colmap_pointcloud:
- Require images.
- Require point cloud file OR existing server pointcloud_path.

### 4.3 Step 3: Validation before create/start

Client-side checks:
- mode is selected.
- pointcloud_only -> dataset_path is not empty.
- hybrid_colmap_pointcloud -> point cloud source is provided.
- image-based modes -> at least one image selected.

Server-side checks still run and remain authoritative.

## 5. End-to-End Flows

## 5.1 Flow A: colmap_images

1. POST /jobs with training_mode=colmap_images
2. POST /jobs/{id}/upload (batch upload)
3. POST /jobs/{id}/upload/complete
4. Poll GET /jobs/{id}/progress or subscribe WS /jobs/{id}/ws
5. GET /jobs/{id}/model when status=completed

Example create payload:

```json
{
  "name": "scene_colmap",
  "training_mode": "colmap_images",
  "auto_start": true,
  "brush_config": {
    "total_steps": 30000,
    "max_resolution": 1024
  }
}
```

## 5.2 Flow B: pointcloud_only

Option 1 (auto-start):
1. POST /jobs with training_mode=pointcloud_only, dataset_path, auto_start=true
2. Monitor progress

Option 2 (manual start):
1. POST /jobs with training_mode=pointcloud_only, dataset_path, auto_start=false
2. POST /jobs/{id}/start with training_mode/dataset_path
3. Monitor progress

Example create payload (auto-start):

```json
{
  "name": "scene_no_colmap",
  "training_mode": "pointcloud_only",
  "dataset_path": "/data/uploads/NewScan",
  "auto_start": true,
  "brush_config": {
    "total_steps": 30000,
    "max_resolution": 1024
  }
}
```

## 5.3 Flow C: hybrid_colmap_pointcloud

1. POST /jobs with training_mode=hybrid_colmap_pointcloud
2. Upload images
3. Upload point cloud (.ply):
   - small/medium file: POST /pointcloud
   - large file: POST /pointcloud/chunk
4. POST /jobs/{id}/upload/complete
5. Monitor progress

Example create payload:

```json
{
  "name": "scene_hybrid",
  "training_mode": "hybrid_colmap_pointcloud",
  "auto_start": false,
  "brush_config": {
    "total_steps": 30000,
    "max_resolution": 1024
  }
}
```

Example start payload (if needed):

```json
{
  "training_mode": "hybrid_colmap_pointcloud"
}
```

## 6. cURL Examples

## 6.1 Create job (mode selection)

```bash
curl -X POST http://SERVER:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_scene",
    "training_mode": "colmap_images",
    "auto_start": true
  }'
```

## 6.2 Upload point cloud for hybrid

```bash
curl -X POST http://SERVER:8080/jobs/JOB_ID/pointcloud \
  -F "file=@points3d.ply"
```

## 6.3 Start pointcloud_only manually

```bash
curl -X POST http://SERVER:8080/jobs/JOB_ID/start \
  -H "Content-Type: application/json" \
  -d '{
    "training_mode": "pointcloud_only",
    "dataset_path": "/data/uploads/NewScan"
  }'
```

## 7. JavaScript Example (fetch)

```javascript
async function createJob(baseUrl, { name, mode, datasetPath }) {
  const payload = {
    name,
    training_mode: mode,
    auto_start: mode !== "pointcloud_only",
    brush_config: { total_steps: 30000, max_resolution: 1024 }
  };

  if (datasetPath) payload.dataset_path = datasetPath;

  const res = await fetch(`${baseUrl}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) throw new Error(`Create failed: ${res.status}`);
  return await res.json();
}

async function uploadPointcloud(baseUrl, jobId, file) {
  const form = new FormData();
  form.append("file", file, file.name);

  const res = await fetch(`${baseUrl}/jobs/${jobId}/pointcloud`, {
    method: "POST",
    body: form
  });

  if (!res.ok) throw new Error(`Pointcloud upload failed: ${res.status}`);
  return await res.json();
}
```

## 8. Error Cases and Client Handling

Common errors:

1. pointcloud_only mode without dataset_path
- Server message: pointcloud_only mode requires dataset_path
- Client action: ask user to set a valid server path.

2. hybrid mode without pointcloud_path/pointcloud upload
- Server message: hybrid_colmap_pointcloud mode requires pointcloud_path
- Client action: block start until .ply is uploaded or path is provided.

3. Invalid point cloud extension
- Server message: Only .ply point cloud is supported
- Client action: restrict file picker to .ply.

4. Job start in invalid state
- Server message: Cannot start job (invalid status)
- Client action: refresh job status and show current state timeline.

## 9. State Expectations by Mode

colmap_images:
- uploaded -> colmap_running -> colmap_done -> training -> completed

pointcloud_only:
- pending/uploaded -> colmap_done (virtual skip) -> training -> completed

hybrid_colmap_pointcloud:
- uploaded -> colmap_running -> colmap_done -> training -> completed
(with external points3d injected between colmap_done and training start)

## 10. Migration Notes for Existing Clients

If your client already uses skip_colmap:
- Existing requests continue to work.
- Recommended migration:
  - replace skip_colmap=true with training_mode=pointcloud_only
  - replace skip_colmap=false with training_mode=colmap_images

For new integrations, use training_mode explicitly.

## 11. Minimal Integration Checklist

1. Add mode selector to UI.
2. Add conditional fields for dataset_path and pointcloud file.
3. Send training_mode in create/start payload.
4. Add hybrid pointcloud upload call.
5. Add robust progress + error UI using /progress or WS.
6. Validate inputs before submission.
7. Keep backward compatibility if old workflows still exist.
8. If you want a video preview, add the preview flow from section 12 - it's opt-in, not part of the checklist above.

## 12. Video Preview API

Renders a short flythrough video of a completed job's trained model, following
the same camera path the scene was actually captured/trained on. Works for
**every training mode** (colmap_images, pointcloud_only, hybrid_colmap_pointcloud,
hybrid_known_poses) - the server picks the right camera poses internally, you
don't need to know or care which mode a job used.

**This is opt-in and on-demand only.** No preview is ever generated
automatically after training - you must explicitly POST to request one, for
every job, every time. There is no "preview_enabled" flag on job creation.

It is a fast approximate renderer (not Brush's real-time viewer), meant for a
quick visual sanity check in your UI, not a substitute for downloading and
viewing the actual `.ply`.

### 12.1 Request a preview: `POST /jobs/{job_id}/preview`

Preconditions: job status must be `completed` (a model must exist). Returns
`409` if a preview render is already in progress for that job - don't retry in
a tight loop, poll status instead (12.2).

Body (all fields optional - defaults are tuned to look like a slow, smooth
flythrough, not a slideshow):

```json
{
  "num_waypoints": 40,
  "sub_steps": 48,
  "path_smoothing": 0.5,
  "fps": 24,
  "resolution": 960
}
```

| Field | Default | Meaning |
|---|---|---|
| `num_waypoints` | 40 | How many real camera poses from the capture are sampled along the path. |
| `sub_steps` | 48 | Smoothly-interpolated frames generated *between* each pair of waypoints. **This controls playback speed, not just frame count** - raising it slows the camera down (more frames covering the same real path) while staying smooth at the same fps. Total frames ≈ `num_waypoints * sub_steps`. Do **not** "speed up" a preview by lowering `fps` instead of raising `sub_steps` - that just makes it choppier at the same speed, not smoother. |
| `path_smoothing` | 0.5 | 0 = fly the raw sampled capture poses as-is (can have sharp turns where the operator's real path zig-zagged); 1 = fully smooth the path. Smoothing is automatically reduced near scene geometry so the smoothed path doesn't cut through objects the original path went around. |
| `fps` | 24 | Output video frame rate. |
| `resolution` | 960 | Output width in pixels; height follows the source image aspect ratio. |

With the defaults, expect ≈`40*48/24` ≈ 80 seconds of video. Lowering
`sub_steps` shortens both the render time and the video length (still smooth,
just a faster-feeling flythrough) - use that instead of touching `fps` if you
want a shorter preview.

Response (immediate, before rendering finishes):

```json
{"job_id": "...", "status": "rendering", "error": null, "download_url": null, "stats": null}
```

### 12.2 Poll status: `GET /jobs/{job_id}/preview`

```json
{
  "job_id": "...",
  "status": "completed",
  "error": null,
  "download_url": "http://SERVER:8080/jobs/{job_id}/preview/download",
  "stats": {
    "num_splats": 1638948,
    "num_frames": 753,
    "fps": 24,
    "duration_seconds": 31.4,
    "resolution": [960, 720],
    "device": "cuda"
  }
}
```

`status` is one of: `not_requested` (never asked for one), `rendering`,
`completed`, `failed` (see `error` for the message). Poll every 5-10s -
render time is roughly proportional to `num_waypoints * sub_steps` and to
model size; budget a few minutes for default settings, not seconds.

### 12.3 Download: `GET /jobs/{job_id}/preview/download`

Returns the `.mp4` file directly (`404` if not completed yet). Use
`download_url` from 12.2 rather than constructing this yourself.

### 12.4 cURL example

```bash
curl -X POST http://SERVER:8080/jobs/JOB_ID/preview \
  -H "Content-Type: application/json" -d '{}'

# poll:
curl http://SERVER:8080/jobs/JOB_ID/preview

# once status=="completed":
curl -o preview.mp4 http://SERVER:8080/jobs/JOB_ID/preview/download
```

### 12.5 JavaScript example (fetch)

```javascript
async function requestPreview(baseUrl, jobId, opts = {}) {
  const res = await fetch(`${baseUrl}/jobs/${jobId}/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts) // {} for defaults
  });
  if (res.status === 409) return; // already rendering, just start polling
  if (!res.ok) throw new Error(`Preview request failed: ${res.status}`);
  return await res.json();
}

async function pollPreview(baseUrl, jobId, { intervalMs = 5000 } = {}) {
  while (true) {
    const res = await fetch(`${baseUrl}/jobs/${jobId}/preview`);
    const info = await res.json();
    if (info.status === "completed") return info; // info.download_url is ready
    if (info.status === "failed") throw new Error(info.error);
    await new Promise(r => setTimeout(r, intervalMs));
  }
}
```

### 12.6 Error cases

1. Job not completed yet → `400 Training not completed - nothing to preview yet`. Only offer the "preview" button once job status is `completed`.
2. Preview already rendering for this job → `409 A preview render is already in progress for this job`. Just resume polling 12.2, don't resubmit.
3. `preview_status == "failed"` → check `error` field. Most likely cause: the job's on-disk data (COLMAP sparse model or prepared dataset) was cleaned up after training completed - re-requesting won't help in that case.

## 13. Camera Positions API

### 13.1 What this is for

Section 12's preview video is good for a quick sanity-check glance, but it's
a rendered `.mp4` - not useful if what you actually need is a **starting
camera position inside a real 3DGS viewer** (Brush's own viewer, superspl.at,
antimatter15's webgl splat viewer, a three.js-based viewer you built, etc.)
that has loaded the job's exported `.ply` directly. Scrolling through the
scan's hundreds of near-duplicate training photos to eyeball "which one was a
good angle" doesn't work either - they're frames of a walkthrough, not
curated viewpoints, and their filenames tell you nothing about where they sit
in 3D space.

`GET /jobs/{job_id}/camera_positions` solves that: it hands back a small,
spatially-diverse set of the *actual capture camera poses*, converted into
plain world-space vectors you can feed directly into whatever camera API your
viewer exposes. No video rendering, no GPU, no polling - it's a synchronous
call that just reads camera pose files and returns in well under a second
even for large jobs.

**Works for every training mode** (`colmap_images`, `pointcloud_only`,
`hybrid_colmap_pointcloud`, `hybrid_known_poses`) - same as the preview video,
the server internally picks the right pose source per job so you never need
to know or care which mode a job used (see section 12's intro / the
`preview_renderer.py` module docstring if you're curious why that distinction
matters at all: COLMAP-only jobs live in COLMAP's own independently-solved
coordinate frame, not the original capture's world frame).

**Important - "well-spread", not "aesthetically best"**: the server has no
concept of which angle looks nicest. Selection is purely geometric (farthest-
point sampling over position - see 13.6), so you get poses spread around the
space that was actually scanned, all guaranteed to be real, valid capture
viewpoints (not interpolated/synthetic). Treat the result as a set of good
*candidates* to jump to, not a single "the best angle" answer - if you're
building a UI, showing a small gallery/dropdown of `count` options (using
`source_image` to cross-reference against the original photos) works well.

### 13.2 Request

```
GET /jobs/{job_id}/camera_positions?count=12
```

| Param | Location | Default | Range | Meaning |
|---|---|---|---|---|
| `job_id` | path | - | - | Must be a job with `status == "completed"`. |
| `count` | query | 12 | 1-100 | How many camera poses to return. |

```bash
curl "http://SERVER:8080/jobs/JOB_ID/camera_positions?count=8"
```

### 13.3 Response

```json
{
  "job_id": "b6675af5-c61c-421f-b6d3-0d68d50b8c60",
  "convention": "Right-handed world space, same frame as the exported .ply's splat positions. `forward`/`up`/`right`/`position` are already expressed in world space - aim a camera at `position` along `forward` with `up` as its up vector, no extra convention knowledge needed. `quaternion_wxyz` (w,x,y,z) represents the same orientation for tools that want a quaternion/matrix instead, using a local axis convention where the camera looks down its own -Z with +Y up.",
  "count": 8,
  "cameras": [
    {
      "position": [2.29, -1.72, 0.55],
      "forward": [-0.87, 0.46, 0.20],
      "up": [0.45, 0.54, 0.71],
      "right": [0.21, 0.70, -0.68],
      "quaternion_wxyz": [0.62, 0.47, 0.62, 0.10],
      "fov_x_deg": 40.7,
      "fov_y_deg": 66.8,
      "source_image": "frame_000013.jpg"
    }
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `position` | `[x,y,z]` | Camera location, world space - same coordinate frame as the vertex positions in the job's exported `.ply`. |
| `forward` | `[x,y,z]`, unit vector | Direction the camera looks, world space. |
| `up` | `[x,y,z]`, unit vector | Camera's up direction, world space. Orthogonal to `forward`. |
| `right` | `[x,y,z]`, unit vector | Camera's right direction, world space. Redundant with `forward`×`up` but included so you don't have to compute it. |
| `quaternion_wxyz` | `[w,x,y,z]` | Same orientation as a quaternion, **only** if you're setting a raw camera transform instead of a look-at (see 13.4.2 for its axis convention - `position`/`forward`/`up` avoid needing to know this at all). |
| `fov_x_deg` / `fov_y_deg` | degrees | The original capture camera's horizontal/vertical field of view. Useful for matching perspective if you want the view to look like the original photo; not required if you just want *a* reasonable viewpoint. |
| `source_image` | string | Filename of the original photo taken from (approximately) this pose - purely informational, e.g. for a UI label or for cross-referencing which part of the scan this angle covers. |

`count` in the response is the number actually returned - it can be smaller
than what you asked for if the job has fewer registered camera frames than
`count`.

### 13.4 Using the response to set up a camera

#### 13.4.1 Look-at style (recommended - works with almost any viewer/engine)

This is the simplest path and needs zero knowledge of this pipeline's
internal conventions, because `position`/`forward`/`up` are already plain
world-space vectors.

**three.js:**
```javascript
const cam = cameras[0]; // one entry from the `cameras` array
camera.position.set(...cam.position);
camera.up.set(...cam.up);
const target = cam.position.map((v, i) => v + cam.forward[i]);
camera.lookAt(...target);
```

**Generic pseudocode (any engine with a `lookAt(eye, target, up)` API):**
```
eye    = cam.position
target = cam.position + cam.forward   // any point along forward works, this just needs to be non-coincident with eye
up     = cam.up
viewer.setCamera(lookAt(eye, target, up))
```

#### 13.4.2 Quaternion style (if your viewer wants a transform, not a look-at)

Use `quaternion_wxyz` together with `position` to build a camera-to-world
transform directly. Its axis convention: **the camera looks down its own
local -Z axis, with +Y as local up** (the same convention used by
three.js/WebGL cameras, glTF, and Nerfstudio's `transforms.json` - if your
viewer is any of those, this quaternion is a drop-in camera rotation, no
conversion needed). If you're not sure which style to use, prefer 13.4.1 -
it sidesteps this entirely.

```javascript
const cam = cameras[0];
const [w, x, y, z] = cam.quaternion_wxyz;
camera.position.set(...cam.position);
camera.quaternion.set(x, y, z, w); // note: three.js Quaternion ctor is (x,y,z,w), not (w,x,y,z)
```

#### 13.4.3 Matching field of view

If you want the rendered view to resemble the original photo's framing
(rather than whatever FOV your viewer defaults to), set your camera's
vertical FOV to `fov_y_deg` - but only if your viewport's aspect ratio
roughly matches the original photo's (a perspective camera's *horizontal*
FOV changes with aspect ratio even at a fixed vertical FOV, so a very
different aspect ratio viewport will crop differently than the original
photo regardless). For most "jump to a good starting angle" use cases you
can just leave your viewer's own default FOV alone and ignore this field.

### 13.5 Error cases

1. Job doesn't exist → `404 Job not found`.
2. Job not completed yet → `400 Training not completed - no camera poses to offer yet`. Only offer this once job status is `completed`, same precondition as the preview video.
3. Job's on-disk camera pose data (COLMAP sparse model / prepared dataset) was cleaned up after training → `400` with a message naming what's missing. Same underlying cause as preview-video error case 3 in 12.6 - re-requesting won't help.

### 13.6 How selection works (if you need to reason about `count`)

Poses are chosen by greedy farthest-point sampling over 3D position: the
first pose is the earliest registered frame, then each subsequent pick is
whichever remaining frame is farthest (in world space) from every pose
already picked. The practical effect is even coverage of the scanned volume
rather than clustering wherever the operator moved slowly or lingered.

- Small `count` (4-8): a handful of maximally-different overview angles - good for a "pick your starting view" gallery.
- Larger `count` (20-50): denser coverage, still spread out - useful if you want to let a user scrub through more fine-grained angle options.
- `count` above the job's actual number of registered frames just returns all of them (no duplicates, no error).
