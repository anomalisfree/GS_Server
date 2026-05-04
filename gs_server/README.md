# Gaussian Splatting Training Server

Automatic server for image processing through COLMAP and Gaussian Splatting training with Brush.

## Features

- 📤 **Image Upload** - accepts images via HTTP API (supports large files with chunked upload)
- 🔄 **EXIF Normalization** - automatically bakes EXIF orientation into pixels before any processing, so COLMAP, Brush and masks all see the same orientation (critical for smartphone photos)
- 🎭 **Semantic Masking** - DeepLabV3 segmentation removes dynamic objects (people, cars, bikes, etc.) before COLMAP
- 🔄 **COLMAP Processing** - automatic Structure-from-Motion reconstruction
- 🎓 **Brush Training** - 3D Gaussian Splatting model training with tuned anti-floater parameters
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
| `COLMAP_USE_GPU` | Use GPU for COLMAP | `true` |
| `MASKING_ENABLED` | Enable semantic masking | `true` |
| `MASKING_REMOVE_CLASSES` | Comma-separated classes to mask | `bicycle,bus,car,cat,dog,motorbike,person` |
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
4. **PROCESSING** (automatic, before COLMAP)
   - **Stage 0a: EXIF Normalization** - bakes EXIF orientation into pixels; ensures COLMAP, Brush, and masks all read identical pixel data regardless of how the phone saved the image
   - **Stage 0b: Semantic Masking** - generates per-image masks using DeepLabV3 (ResNet-101, VOC) to exclude dynamic objects
5. **COLMAP_RUNNING** - COLMAP processing
   - Feature Extraction
   - Feature Matching
   - Sparse Reconstruction
   - Image Undistortion
6. **COLMAP_DONE** - COLMAP completed
7. **TRAINING** - Brush training
8. **COMPLETED** - Done!

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

## Masking Settings

Masking uses DeepLabV3-ResNet101 (COCO/VOC) to detect and mask out dynamic objects so they do not corrupt COLMAP point cloud and Gaussian Splatting training.

Supported classes (VOC): `aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`

```json
{
    "masking_config": {
        "enabled": true,
        "remove_classes": ["bicycle", "bus", "car", "cat", "dog", "motorbike", "person"]
    }
}
```

## Brush Settings

The defaults are tuned to minimise floaters and constrain Gaussian growth to well-covered scene areas.

```json
{
    "brush_config": {
        "total_steps": 30000,
        "max_resolution": 1920,
        "eval_every": 1000,
        "export_every": 2500,
        "lr_mean": 2e-5,
        "lr_opac": 0.012,

        "ssim_weight": 0.3,

        "max_splats": 2000000,
        "growth_grad_threshold": 0.006,
        "growth_select_fraction": 0.12,
        "stop_growth_at": 12000,

        "opac_decay": 0.007,
        "scale_decay": 0.004
    }
}
```

### Anti-floater parameters explained

| Parameter | Default | Description |
|---|---|---|
| `growth_grad_threshold` | `0.006` | Gaussians only grow where position gradient exceeds this value — i.e. in regions well-covered by COLMAP points. Raise to be more conservative. |
| `growth_select_fraction` | `0.12` | Fraction of above-threshold Gaussians that actually grow each refinement step. Lower = fewer floaters. |
| `stop_growth_at` | `12000` | Stop densification at this step. After this only decay and pruning happen. |
| `max_splats` | `2000000` | Hard cap on total Gaussians. Limits noise budget. |
| `opac_decay` | `0.007` | Opacity regularisation — transparent/unused Gaussians are pruned faster. |
| `scale_decay` | `0.004` | Scale regularisation — prevents Gaussians from bloating into empty space. |
| `ssim_weight` | `0.3` | Weight of structural similarity loss vs. L1. Higher = better structure preservation. |

## License

MIT
