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
