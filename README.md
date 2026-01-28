# GS Server - Gaussian Splatting Training Server

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

Automated server for processing images via COLMAP and training 3D Gaussian Splatting models using [Brush](https://github.com/ArthurBrussee/brush).

## 🚀 Features

- **📤 Image Upload** - REST API for uploading images (supports chunked upload for large files)
- **🔄 COLMAP Processing** - Automatic Structure-from-Motion reconstruction
- **🎓 Brush Training** - 3D Gaussian Splatting model training
- **📊 Real-time Monitoring** - WebSocket progress tracking
- **📥 Model Download** - Download trained .ply models
- **🔌 Multi-client Support** - Python, C#/Unity, JavaScript clients included

## 📋 Requirements

### System Requirements
- Windows 10/11 or Linux
- Python 3.10+
- CUDA-compatible GPU (strongly recommended)
- 16+ GB RAM
- 50+ GB free disk space

### Dependencies
- [COLMAP](https://colmap.github.io/) - Structure from Motion
- [Brush](https://github.com/ArthurBrussee/brush) - Gaussian Splatting training
- [Rust](https://rustup.rs/) (for building Brush)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anomalisfree/GS_Server.git
cd GS_Server
```

### 2. Install COLMAP

#### Windows
Download from [COLMAP releases](https://github.com/colmap/colmap/releases) and extract to `colmap/` folder:

```
GS_Server/
├── colmap/
│   ├── bin/
│   │   └── colmap.exe
│   └── ...
```

#### Linux
```bash
sudo apt-get install colmap
```

### 3. Install Brush

```bash
# Clone Brush (already included or clone separately)
git clone https://github.com/ArthurBrussee/brush.git brush

# Build Brush (requires Rust)
cd brush
cargo build --release
cd ..
```

### 4. Install Python Dependencies

```bash
cd gs_server
pip install -r requirements.txt
```

### 5. Configure Environment (Optional)

Create a `.env` file or set environment variables:

```bash
# Server settings
GS_HOST=0.0.0.0
GS_PORT=8080

# Paths (optional - uses defaults if not set)
GS_COLMAP_EXE=../colmap/bin/colmap.exe
GS_BRUSH_DIR=../brush
GS_JOBS_DIR=../jobs

# Training settings
COLMAP_USE_GPU=true
BRUSH_TOTAL_STEPS=30000
BRUSH_MAX_RESOLUTION=1920
```

## 🚀 Running the Server

### Windows

```bash
# Using batch file
start_server.bat

# Or manually
python -m gs_server
```

### Linux/Mac

```bash
python -m gs_server
```

### With Custom Settings

```bash
GS_HOST=0.0.0.0 GS_PORT=8080 python -m gs_server
```

### Running in Background (Windows)

```bash
start_server_background.bat
```

## 📖 API Documentation

After starting the server, access the interactive API documentation at:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Quick API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Server status |
| `/jobs` | POST | Create new job |
| `/jobs` | GET | List all jobs |
| `/jobs/{id}` | GET | Get job info |
| `/jobs/{id}/upload` | POST | Upload images |
| `/jobs/{id}/upload/complete` | POST | Complete upload & start processing |
| `/jobs/{id}/progress` | GET | Get progress |
| `/jobs/{id}/model` | GET | Download trained model |
| `/jobs/{id}/ws` | WS | WebSocket progress updates |

## 💻 Client Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8080"

# 1. Create job
job = requests.post(f"{BASE_URL}/jobs", json={
    "name": "my_scene",
    "auto_start": True
}).json()
job_id = job["job_id"]

# 2. Upload images
files = [("files", open(f"image{i}.jpg", "rb")) for i in range(10)]
requests.post(f"{BASE_URL}/jobs/{job_id}/upload", files=files)

# 3. Complete upload
requests.post(f"{BASE_URL}/jobs/{job_id}/upload/complete")

# 4. Check progress
progress = requests.get(f"{BASE_URL}/jobs/{job_id}/progress").json()
print(f"Progress: {progress['overall_progress']}%")

# 5. Download model when complete
response = requests.get(f"{BASE_URL}/jobs/{job_id}/model")
with open("model.ply", "wb") as f:
    f.write(response.content)
```

### cURL

```bash
# Create job
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{"name": "my_scene", "auto_start": true}'

# Upload images
curl -X POST http://localhost:8080/jobs/JOB_ID/upload \
  -F "files=@image1.jpg" -F "files=@image2.jpg"

# Complete upload
curl -X POST http://localhost:8080/jobs/JOB_ID/upload/complete

# Get progress
curl http://localhost:8080/jobs/JOB_ID/progress

# Download model
curl -o model.ply http://localhost:8080/jobs/JOB_ID/model
```

See [CLIENT_INTEGRATION_GUIDE.md](CLIENT_INTEGRATION_GUIDE.md) for complete examples in Python, C#/Unity, and JavaScript.

## 📁 Project Structure

```
GS_Server/
├── gs_server/              # Main server package
│   ├── __init__.py
│   ├── __main__.py        # Entry point
│   ├── app.py             # FastAPI application
│   ├── config.py          # Configuration
│   ├── models.py          # Data models
│   ├── job_manager.py     # Job queue management
│   ├── colmap_runner.py   # COLMAP pipeline
│   ├── brush_runner.py    # Brush training
│   └── requirements.txt   # Python dependencies
├── colmap/                 # COLMAP binaries
├── brush/                  # Brush source
├── jobs/                   # Job working directories
├── logs/                   # Server logs
├── CLIENT_INTEGRATION_GUIDE.md  # Client integration guide
├── start_server.bat       # Windows startup script
├── start_server_background.bat
├── stop_server.bat
└── README.md
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GS_HOST` | Server host | `0.0.0.0` |
| `GS_PORT` | Server port | `8080` |
| `GS_MAX_UPLOAD_GB` | Max upload size (GB) | `50.0` |
| `GS_JOBS_DIR` | Jobs directory | `./jobs` |
| `GS_COLMAP_EXE` | Path to COLMAP | `../colmap/bin/colmap.exe` |
| `GS_BRUSH_DIR` | Path to Brush | `../brush` |
| `COLMAP_USE_GPU` | Use GPU for COLMAP | `true` |
| `BRUSH_TOTAL_STEPS` | Training steps | `30000` |
| `BRUSH_MAX_RESOLUTION` | Max image resolution | `1920` |
| `BRUSH_EVAL_EVERY` | Evaluation interval | `1000` |
| `BRUSH_EXPORT_EVERY` | Model export interval | `2500` |

### Training Parameters

Pass via API when creating a job:

```json
{
    "name": "my_scene",
    "auto_start": true,
    "brush_config": {
        "total_steps": 30000,
        "max_resolution": 1920,
        "eval_every": 1000,
        "export_every": 5000,
        "ssim_weight": 0.2
    }
}
```

## 📊 Processing Pipeline

1. **Upload** → Images uploaded to job directory
2. **COLMAP** → Structure from Motion reconstruction
   - Feature extraction
   - Feature matching
   - Sparse reconstruction
   - Image undistortion
3. **Training** → Brush 3D Gaussian Splatting
   - Model initialization
   - Iterative optimization
   - Periodic evaluation (PSNR, SSIM)
   - Model export (.ply)
4. **Download** → Retrieve trained model

## 🔍 Monitoring Progress

### REST API

```bash
curl http://localhost:8080/jobs/JOB_ID/progress
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/jobs/JOB_ID/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`Progress: ${data.data.overall_progress}%`);
};
```

## 📝 Recommendations for Best Results

1. **Number of images**: 50-500 photos
2. **Resolution**: 1080p to 4K
3. **Quality**: Sharp, well-lit photos
4. **Coverage**: 60-80% overlap between images
5. **Format**: JPG or PNG (RGB)
6. **Movement**: Capture object from all angles

## 🐛 Troubleshooting

### COLMAP Fails

- Check GPU drivers are up to date
- Ensure images have sufficient overlap
- Try with `COLMAP_USE_GPU=false` for CPU mode

### Brush Training Issues

- Ensure Rust/Cargo is installed
- Check CUDA drivers for GPU training
- Monitor GPU memory usage

### Server Won't Start

- Check port 8080 is not in use
- Verify Python 3.10+ is installed
- Install missing dependencies: `pip install -r requirements.txt`

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Brush](https://github.com/ArthurBrussee/brush) - 3D Gaussian Splatting implementation
- [COLMAP](https://colmap.github.io/) - Structure from Motion
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
