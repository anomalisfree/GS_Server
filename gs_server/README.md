# Gaussian Splatting Training Server

Automatic server for image processing through COLMAP and Gaussian Splatting training with Brush.

## Features

- 📤 **Image Upload** - accepts images via HTTP API (supports large files with chunked upload)
- 🔄 **COLMAP Processing** - automatic Structure-from-Motion reconstruction
- 🎓 **Brush Training** - 3D Gaussian Splatting model training
- 📊 **Real-time Monitoring** - progress tracking via WebSocket
- 📥 **Model Download** - retrieve ready .ply files

## Requirements

- Python 3.10+
- COLMAP (installed in `../colmap`)
- Brush (installed in `../brush`)
- CUDA-compatible GPU (recommended)

## Installation

```bash
cd gs_server
pip install -r requirements.txt
```

## Running

```bash
# Basic run
python -m gs_server

# With parameters
GS_HOST=0.0.0.0 GS_PORT=8080 python -m gs_server
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GS_HOST` | Server host | `0.0.0.0` |
| `GS_PORT` | Server port | `8080` |
| `GS_MAX_UPLOAD_GB` | Max upload size (GB) | `50.0` |
| `GS_JOBS_DIR` | Jobs directory | `./jobs` |
| `GS_COLMAP_EXE` | Path to COLMAP | `../colmap/bin/colmap.exe` |
| `GS_BRUSH_DIR` | Path to Brush | `../brush` |
| `COLMAP_USE_GPU` | Use GPU | `true` |
| `BRUSH_TOTAL_STEPS` | Training steps | `30000` |
| `BRUSH_MAX_RESOLUTION` | Max resolution | `1920` |

## API Endpoints

### Information

- `GET /` - Server info
- `GET /status` - Server status
- `GET /docs` - Swagger documentation

### Jobs

- `POST /jobs` - Create a new job
- `GET /jobs` - List jobs
- `GET /jobs/{job_id}` - Job information
- `GET /jobs/{job_id}/progress` - Job progress
- `DELETE /jobs/{job_id}` - Delete job
- `POST /jobs/{job_id}/cancel` - Cancel job
- `POST /jobs/{job_id}/start` - Start manually

### Upload

- `POST /jobs/{job_id}/upload` - Upload images
- `POST /jobs/{job_id}/upload/complete` - Complete upload
- `POST /jobs/{job_id}/upload/chunk` - Chunked upload

### Download

- `GET /jobs/{job_id}/model` - Download model
- `GET /jobs/{job_id}/model/info` - Model information
- `GET /jobs/{job_id}/exports` - List all exports
- `GET /jobs/{job_id}/exports/{filename}` - Download specific export

### WebSocket

- `WS /jobs/{job_id}/ws` - Real-time progress updates

## Usage Example (Python client)

```python
import requests
import time

BASE_URL = "http://localhost:8080"

# 1. Create job
response = requests.post(f"{BASE_URL}/jobs", json={
    "name": "my_scene",
    "auto_start": True,
    "brush_config": {
        "total_steps": 30000,
        "max_resolution": 1920
    }
})
job = response.json()
job_id = job["job_id"]
print(f"Created job: {job_id}")

# 2. Upload images
images = ["image1.jpg", "image2.jpg", ...]  # List of file paths
files = [("files", open(img, "rb")) for img in images]
requests.post(f"{BASE_URL}/jobs/{job_id}/upload", files=files)

# 3. Complete upload (starts processing)
requests.post(f"{BASE_URL}/jobs/{job_id}/upload/complete")

# 4. Track progress
while True:
    progress = requests.get(f"{BASE_URL}/jobs/{job_id}/progress").json()
    print(f"Status: {progress['status']}, Progress: {progress['overall_progress']:.1f}%")
    
    if progress["status"] in ["completed", "failed", "cancelled"]:
        break
    time.sleep(5)

# 5. Download model
if progress["status"] == "completed":
    response = requests.get(f"{BASE_URL}/jobs/{job_id}/model")
    with open("model.ply", "wb") as f:
        f.write(response.content)
    print("Model downloaded!")
```

## WebSocket Client Example

```python
import asyncio
import websockets
import json

async def monitor_job(job_id):
    uri = f"ws://localhost:8080/jobs/{job_id}/ws"
    
    async with websockets.connect(uri) as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            
            if data["type"] == "progress":
                progress = data["data"]
                print(f"Progress: {progress['overall_progress']:.1f}%")
                
                if progress.get("training"):
                    t = progress["training"]
                    print(f"  Step: {t['current_step']}/{t['total_steps']}")
                    if t.get("psnr"):
                        print(f"  PSNR: {t['psnr']:.2f}")
            
            elif data["type"] == "completed":
                print("Training completed!")
                break
            
            elif data["type"] == "error":
                print(f"Error: {data['data']['error']}")
                break

asyncio.run(monitor_job("your-job-id"))
```

## Directory Structure

```
gs_server/
├── jobs/                    # Job working directories
│   └── {job_id}/
│       ├── state.json       # Job state
│       ├── images/          # Uploaded images
│       ├── colmap/          # COLMAP results
│       │   ├── database.db
│       │   ├── sparse/
│       │   └── dense/
│       └── output/          # Training results
│           └── model_*.ply
├── uploads/                 # Temporary files
└── models/                  # Ready models
```

## Processing Stages

1. **PENDING** - Job created, waiting for upload
2. **UPLOADING** - Image upload in progress
3. **UPLOADED** - Upload completed
4. **COLMAP_RUNNING** - COLMAP processing
   - Feature Extraction
   - Feature Matching
   - Sparse Reconstruction
   - Image Undistortion
5. **COLMAP_DONE** - COLMAP completed
6. **TRAINING** - Brush training
7. **COMPLETED** - Done!

## COLMAP Settings

```json
{
    "colmap_config": {
        "use_gpu": true,
        "camera_model": "OPENCV",
        "single_camera": true,
        "max_image_size": 3200,
        "matcher_type": "exhaustive"
    }
}
```

## Brush Settings

```json
{
    "brush_config": {
        "total_steps": 30000,
        "max_resolution": 1920,
        "eval_every": 1000,
        "export_every": 5000,
        "lr_mean": 2e-5,
        "lr_opac": 0.012,
        "ssim_weight": 0.2,
        "max_splats": 10000000
    }
}
```

## License

MIT
