"""
Gaussian Splatting Server Configuration
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    """Basic server settings"""
    host: str = "0.0.0.0"
    port: int = 8080
    max_upload_size_gb: float = 50.0  # Максимальный размер загрузки в ГБ
    
    
@dataclass
class PathsConfig:
    """Paths to directories and executables"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Рабочие директории
    jobs_dir: Path = field(default=None)
    uploads_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    
    # Пути к инструментам
    colmap_exe: Path = field(default=None)
    brush_dir: Path = field(default=None)
    
    def __post_init__(self):
        if self.jobs_dir is None:
            self.jobs_dir = self.base_dir / "jobs"
        if self.uploads_dir is None:
            self.uploads_dir = self.base_dir / "uploads"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.colmap_exe is None:
            self.colmap_exe = self.base_dir / "colmap" / "bin" / "colmap.exe"
        if self.brush_dir is None:
            self.brush_dir = self.base_dir / "brush"
            
    def ensure_directories(self):
        """Create all required directories"""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ColmapConfig:
    """COLMAP settings"""
    use_gpu: bool = True
    camera_model: str = "OPENCV"  # SIMPLE_PINHOLE, PINHOLE, OPENCV, etc.
    single_camera: bool = True  # Все изображения с одной камеры
    max_image_size: int = 3200  # Максимальный размер изображения
    num_threads: int = -1  # -1 = авто
    
    # Настройки feature extraction
    sift_max_features: int = 8192
    sift_first_octave: int = -1
    
    # Настройки matching
    matcher_type: str = "exhaustive"  # exhaustive, sequential, spatial, vocab_tree
    
    
@dataclass  
class BrushConfig:
    """Brush training settings"""
    total_steps: int = 30000
    max_resolution: int = 1920
    eval_every: int = 1000  # Каждые 1000 шагов выводятся PSNR и SSIM
    export_every: int = 2500  # Экспорт каждые 2500 шагов для более частого обновления
    
    # Learning rates
    lr_mean: float = 2e-5
    lr_mean_end: float = 2e-7
    lr_coeffs_dc: float = 2e-3
    lr_opac: float = 0.012
    lr_scale: float = 7e-3
    lr_scale_end: float = 5e-3
    lr_rotation: float = 2e-3
    
    # Refinement
    refine_every: int = 200  # Каждые 200 шагов выводится количество сплатов
    growth_grad_threshold: float = 0.003
    growth_select_fraction: float = 0.2
    stop_growth_at: int = 15000
    max_splats: int = 10000000
    
    # Loss
    ssim_weight: float = 0.2
    
    # Rerun visualization (отключено по умолчанию для headless)
    rerun_enabled: bool = False


@dataclass
class MaskingConfig:
    """DeepLabV3 segmentation masking settings"""
    enabled: bool = True  # Включить генерацию масок перед COLMAP
    remove_classes: list = field(default_factory=lambda: ["sky", "person"])
    
    
@dataclass
class AppConfig:
    """Full application configuration"""
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    brush: BrushConfig = field(default_factory=BrushConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    

def load_config() -> AppConfig:
    """Load configuration from environment variables"""
    config = AppConfig()
    
    # Server
    config.server.host = os.getenv("GS_HOST", config.server.host)
    config.server.port = int(os.getenv("GS_PORT", config.server.port))
    config.server.max_upload_size_gb = float(os.getenv("GS_MAX_UPLOAD_GB", config.server.max_upload_size_gb))
    
    # Paths
    if base := os.getenv("GS_BASE_DIR"):
        config.paths.base_dir = Path(base)
    if jobs := os.getenv("GS_JOBS_DIR"):
        config.paths.jobs_dir = Path(jobs)
    if uploads := os.getenv("GS_UPLOADS_DIR"):
        config.paths.uploads_dir = Path(uploads)
    if models := os.getenv("GS_MODELS_DIR"):
        config.paths.models_dir = Path(models)
    if colmap := os.getenv("GS_COLMAP_EXE"):
        config.paths.colmap_exe = Path(colmap)
    if brush := os.getenv("GS_BRUSH_DIR"):
        config.paths.brush_dir = Path(brush)
        
    # COLMAP
    config.colmap.use_gpu = os.getenv("COLMAP_USE_GPU", "true").lower() == "true"
    config.colmap.camera_model = os.getenv("COLMAP_CAMERA_MODEL", config.colmap.camera_model)
    config.colmap.max_image_size = int(os.getenv("COLMAP_MAX_IMAGE_SIZE", config.colmap.max_image_size))
    
    # Brush  
    config.brush.total_steps = int(os.getenv("BRUSH_TOTAL_STEPS", config.brush.total_steps))
    config.brush.max_resolution = int(os.getenv("BRUSH_MAX_RESOLUTION", config.brush.max_resolution))
    config.brush.eval_every = int(os.getenv("BRUSH_EVAL_EVERY", config.brush.eval_every))
    config.brush.export_every = int(os.getenv("BRUSH_EXPORT_EVERY", config.brush.export_every))
    
    # Masking
    config.masking.enabled = os.getenv("MASKING_ENABLED", "true").lower() == "true"
    if remove := os.getenv("MASKING_REMOVE_CLASSES"):
        config.masking.remove_classes = [c.strip() for c in remove.split(",")]
    
    config.paths.ensure_directories()
    
    return config


# Глобальный конфиг
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get global configuration"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
