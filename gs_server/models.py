"""
Data models for API
"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class JobStatus(str, Enum):
    """Job statuses"""
    PENDING = "pending"           # Ожидает в очереди
    UPLOADING = "uploading"       # Загрузка файлов
    UPLOADED = "uploaded"         # Файлы загружены
    COLMAP_RUNNING = "colmap_running"   # COLMAP обрабатывает
    COLMAP_DONE = "colmap_done"         # COLMAP завершен
    COLMAP_FAILED = "colmap_failed"     # COLMAP ошибка
    TRAINING = "training"         # Тренировка Brush
    TRAINING_DONE = "training_done"   # Тренировка завершена
    TRAINING_FAILED = "training_failed" # Ошибка тренировки
    COMPLETED = "completed"       # Всё готово
    FAILED = "failed"             # Общая ошибка
    CANCELLED = "cancelled"       # Отменено


class ColmapStage(str, Enum):
    """COLMAP stages"""
    FEATURE_EXTRACTION = "feature_extraction"
    FEATURE_MATCHING = "feature_matching"  
    SPARSE_RECONSTRUCTION = "sparse_reconstruction"
    IMAGE_UNDISTORTION = "image_undistortion"


class TrainingMode(str, Enum):
    """Training mode selection"""
    COLMAP_IMAGES = "colmap_images"
    POINTCLOUD_ONLY = "pointcloud_only"
    HYBRID_COLMAP_POINTCLOUD = "hybrid_colmap_pointcloud"
    HYBRID_KNOWN_POSES = "hybrid_known_poses"


class TrainingProgress(BaseModel):
    """Training progress"""
    current_step: int = 0
    total_steps: int = 0
    steps_per_second: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    current_loss: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    splat_count: int = 0
    last_export_step: int = 0
    last_export_path: Optional[str] = None


class ColmapProgress(BaseModel):
    """COLMAP progress"""
    stage: ColmapStage = ColmapStage.FEATURE_EXTRACTION
    stage_progress: float = 0.0  # 0-100
    images_processed: int = 0
    total_images: int = 0
    features_extracted: int = 0
    matches_found: int = 0
    registered_images: int = 0
    points_3d: int = 0
    message: str = ""


class JobProgress(BaseModel):
    """Overall job progress"""
    overall_progress: float = 0.0  # 0-100
    status: JobStatus = JobStatus.PENDING
    colmap: Optional[ColmapProgress] = None
    training: Optional[TrainingProgress] = None
    message: str = ""
    error: Optional[str] = None


class JobInfo(BaseModel):
    """Job information"""
    job_id: str
    name: str
    status: JobStatus
    progress: JobProgress
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Статистика
    images_count: int = 0
    total_size_bytes: int = 0
    
    # Результаты
    model_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    
    # Конфигурация
    config: Dict[str, Any] = Field(default_factory=dict)


class CreateJobRequest(BaseModel):
    """Request to create a job"""
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None
    
    # Переопределения настроек (опционально)
    colmap_config: Optional[Dict[str, Any]] = None
    brush_config: Optional[Dict[str, Any]] = None
    training_mode: Optional[TrainingMode] = Field(
        default=None,
        description=(
            "Training mode. colmap_images = full COLMAP+Brush, "
            "pointcloud_only = train directly from prepared dataset_path, "
            "hybrid_colmap_pointcloud = COLMAP-estimated cameras + external points3d.ply, "
            "hybrid_known_poses = use dataset_path's known poses directly (no COLMAP pose "
            "estimation), triangulate additional points via COLMAP and merge with the "
            "dataset's own points3d.ply — for scenes where COLMAP can't reliably estimate "
            "poses from images alone (e.g. low-texture surfaces) but accurate poses are "
            "already known from another source (e.g. LiDAR/IMU tracking)"
        )
    )
    pointcloud_path: Optional[str] = Field(
        default=None,
        description="Server-side absolute path to external points3d.ply (for hybrid mode)"
    )
    skip_colmap: bool = False
    dataset_path: Optional[str] = Field(
        default=None,
        description="Server-side path to prepared Nerfstudio/COLMAP-compatible dataset (required for pointcloud_only)"
    )
    
    # Автозапуск после загрузки
    auto_start: bool = True

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "scene_colmap",
                    "training_mode": "colmap_images",
                    "auto_start": True,
                    "brush_config": {
                        "total_steps": 30000,
                        "max_resolution": 1024
                    }
                },
                {
                    "name": "scene_pointcloud_only",
                    "training_mode": "pointcloud_only",
                    "dataset_path": "/data/uploads/NewScan",
                    "auto_start": True,
                    "brush_config": {
                        "total_steps": 30000,
                        "max_resolution": 1024
                    }
                },
                {
                    "name": "scene_hybrid",
                    "training_mode": "hybrid_colmap_pointcloud",
                    "auto_start": False,
                    "pointcloud_path": "/data/shared/points3d_downsampled.ply",
                    "brush_config": {
                        "total_steps": 30000,
                        "max_resolution": 1024
                    }
                }
            ]
        }
    )


class CreateJobResponse(BaseModel):
    """Response for job creation"""
    job_id: str
    upload_url: str
    websocket_url: str


class JobListResponse(BaseModel):
    """List of jobs"""
    jobs: List[JobInfo]
    total: int
    

class UploadProgressResponse(BaseModel):
    """Upload progress"""
    job_id: str
    files_uploaded: int
    total_files: int
    bytes_uploaded: int
    total_bytes: int
    current_file: Optional[str] = None


class StartTrainingRequest(BaseModel):
    """Request to start training"""
    training_mode: Optional[TrainingMode] = Field(
        default=None,
        description="Optional mode override before manual start"
    )
    pointcloud_path: Optional[str] = Field(
        default=None,
        description="Optional server-side points3d.ply path override (used by hybrid mode)"
    )
    skip_colmap: bool = False  # Пропустить COLMAP если данные уже готовы
    dataset_path: Optional[str] = Field(
        default=None,
        description="Optional prepared dataset path override (used by pointcloud_only mode)"
    )
    brush_config: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "training_mode": "pointcloud_only",
                    "dataset_path": "/data/uploads/NewScan"
                },
                {
                    "training_mode": "hybrid_colmap_pointcloud",
                    "pointcloud_path": "/data/jobs/<job_id>/inputs/points3d.ply"
                },
                {
                    "brush_config": {
                        "total_steps": 20000,
                        "max_resolution": 1280
                    }
                }
            ]
        }
    )


class ModelInfo(BaseModel):
    """Information about the trained model"""
    job_id: str
    model_path: str
    model_size_bytes: int
    format: str = "ply"
    training_steps: int
    final_psnr: Optional[float] = None
    final_ssim: Optional[float] = None
    splat_count: int
    created_at: datetime
    download_url: str


class ServerStatus(BaseModel):
    """Server status"""
    status: str = "running"
    version: str
    active_jobs: int
    queued_jobs: int
    completed_jobs: int
    gpu_available: bool
    colmap_version: Optional[str] = None
    brush_version: Optional[str] = None


class WebSocketMessage(BaseModel):
    """WebSocket message"""
    type: str  # progress, log, error, completed
    job_id: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
