"""
Gaussian Splatting Server Configuration
"""

import os
import sys
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
    brush_exe: Optional[Path] = field(default=None)  # Прямой путь к brush binary
    
    def __post_init__(self):
        is_windows = sys.platform == "win32"
        if self.jobs_dir is None:
            self.jobs_dir = self.base_dir / "jobs"
        if self.uploads_dir is None:
            self.uploads_dir = self.base_dir / "uploads"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.colmap_exe is None:
            if is_windows:
                self.colmap_exe = self.base_dir / "colmap" / "bin" / "colmap.exe"
            else:
                # On Linux, prefer system-installed colmap
                self.colmap_exe = Path("/usr/bin/colmap")
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
    camera_model: str = "SIMPLE_RADIAL"  # SIMPLE_PINHOLE, PINHOLE, OPENCV, SIMPLE_RADIAL, etc.
    single_camera: bool = False  # Не фиксировать intrinsics на весь набор
    max_image_size: int = 4500  # Максимальный размер изображения
    num_threads: int = -1  # -1 = авто
    
    # Настройки feature extraction
    sift_max_features: int = 32768
    sift_first_octave: int = -1
    
    # Настройки matching
    matcher_type: str = "exhaustive"  # exhaustive, sequential, spatial, vocab_tree
    # Only used when matcher_type == sequential: enable vocab-tree loop closure
    # detection for scenes where the capture path crosses itself.
    sequential_loop_detection: bool = False
    vocab_tree_path: Optional[Path] = None
    
    
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
    # Higher threshold → grow only where gradient is strong (well-covered by COLMAP points)
    growth_grad_threshold: float = 0.006
    # Lower fraction → less aggressive densification, fewer floaters
    growth_select_fraction: float = 0.12
    # Stop growth earlier → no new Gaussians spawned in poorly-covered areas late in training
    stop_growth_at: int = 12000
    # Lower cap → fewer total Gaussians, less noise budget
    max_splats: int = 2000000
    
    # Decay regularization — maps to Brush's --opac-loss-weight / --scale-loss-weight.
    # NOTE: these were never actually forwarded to the brush process until a bug fix
    # (brush_runner.py read them from config but the values were dropped before the cmd
    # was built), so every job trained so far had actually used Brush's own defaults
    # (1e-9 / 1e-8) regardless of what was configured here. A calibration sweep
    # (3x/10x/30x/100x Brush defaults) on job core_propellers_2 showed this lever does
    # NOT reduce anisotropy of surviving Gaussians (median aniso rose from 391 at 1x to
    # 719 at 30x) — it just prunes splat count indiscriminately (30x: -77%, 100x: -94%)
    # and forces survivors to bloat in size to cover the resulting gaps (maxdim p99 grew
    # 8x at 100x vs 1x), which is what read as "transparent and blurry" to the eye.
    # Reverted to Brush's own defaults; the silhouette-needle problem needs a different
    # fix (targeted post-training anisotropy filtering, not a global loss term).
    opac_decay: float = 1e-9
    scale_decay: float = 1e-8

    # Real gradient-based anisotropy regularization (Brush's own loss, added directly to
    # the training loop - see brush/crates/brush-train/src/train.rs), NOT another global
    # decay term like opac_decay/scale_decay above. Those shrink every splat's magnitude
    # uniformly regardless of shape, which is exactly why the comment above found they
    # "just prune splat count indiscriminately" without touching anisotropy. This instead
    # penalizes the max/min scale-axis RATIO, and only beyond aniso_reg_ratio_threshold -
    # ordinary anisotropic splats (flat walls/floors, 10-50x) are untouched, only the
    # extreme tail is pushed back. 0 = disabled (opt-in; untested at scale yet).
    aniso_reg_weight: float = 0.0
    aniso_reg_ratio_threshold: float = 50.0

    # Depth supervision (Brush's own loss - see brush/crates/brush-train/src/train.rs
    # and brush-render's new depth render channel). Compares the renderer's rendered
    # depth against a ground-truth depth map placed in a sibling "depth/" directory
    # next to "images/" (same auto-discovery convention as "masks/"). Only meaningful
    # for datasets that actually have per-view depth (e.g. LiDAR/ARKit) - 0 disables it.
    depth_loss_weight: float = 0.0
    # Beyond this range (meters), LiDAR/ARKit depth tends to silently return a
    # plausible-looking but wrong value rather than failing - first real test run
    # (no cutoff) visibly corrupted far surfaces (a domed ceiling lost most of its
    # geometry/color) while not fixing the artifact it was meant to help with.
    # Matches Apple's own ~5m LiDAR scanner spec.
    depth_max_range: float = 5.0

    # Position anchor (Brush's own loss - see train.rs): pulls each Gaussian's mean
    # back toward the point-cloud position it (or its splitting ancestor) was
    # created at, once it drifts beyond position_anchor_max_dist. A direct 3D prior,
    # no camera rendering needed - unlike depth_loss_weight this works for any
    # dataset that has an initializing point cloud (pointcloud_only/
    # hybrid_known_poses), not just ones with per-view depth maps. 0 disables it.
    position_anchor_weight: float = 0.0
    position_anchor_max_dist: float = 0.2

    # Photo-consistency tracking (Brush's own accumulator - see train.rs):
    # measures the real rendering residual at each splat's own projected
    # screen position, averaged across every training view it's visible in
    # from photo_consistency_warmup_steps onward. Purely additive/diagnostic -
    # never enters the training loss, only exported alongside the model (as
    # extra photo_resid_avg/photo_resid_count PLY properties) for
    # filter_photometric_inconsistency (floater_filter.py) to consume. Built
    # after position_anchor and aniso_reg (both training-time) and three
    # different post-hoc geometric filters all failed to isolate the
    # AU_Las_Vegas floating-tile artifact - see that investigation report.
    # False (default) adds zero training overhead and produces byte-identical
    # PLY output to before this option existed.
    photo_consistency_enabled: bool = False
    photo_consistency_warmup_steps: int = 15000

    # Loss
    ssim_weight: float = 0.3

    # Rerun visualization (отключено по умолчанию для headless)
    rerun_enabled: bool = False

    # Post-training floater filter (see floater_filter.py). Visual testing on
    # core_propellers_2 (job 210ca694) showed the actual visible "glare" artifact
    # was large, only mildly-anisotropic bright blobs (aniso as low as ~10), NOT
    # the extreme-anisotropy needles this was originally designed for — an
    # aniso+size AND filter missed that population entirely (all 3 initial
    # variants had zero visible effect). Switched to a pure size cut (aniso
    # threshold effectively disabled at 1.01), which visibly removed the glare.
    # Separately: a distance-to-source-scan-cloud check was tried to catch the
    # remaining silhouette-edge "needle" spikes and also had no visible effect —
    # those are believed to be photometric floaters from blended edge pixels
    # (a training-time phenomenon), not geometric outliers; see stop_growth_at.
    #
    # NOTE: the size cutoff is relative (size_multiplier x this model's own
    # median splat size), not absolute. A fixed 10cm cutoff was tuned against a
    # densely-initialized (LiDAR) model (median splat ~0.007) and looked fine
    # there, but on a sparsely-initialized colmap_images model (median splat
    # ~0.13, since fewer points have to cover the same scene) that same absolute
    # cutoff sat *below* the median and threw away 59% of the model. 15x median
    # reproduces the original ~0.5% removal rate on the reference model.
    floater_filter_enabled: bool = True
    floater_filter_aniso_threshold: float = 1.01
    floater_filter_size_multiplier: float = 15.0

    # Second, separate post-training filter pass: removes scattered debris
    # clusters spatially disconnected from the main scanned structure (see
    # floater_filter.filter_isolated_clusters). The needle filter above only
    # looks at a splat's own shape (anisotropy/size) and has no notion of
    # spatial isolation, so it never catches this population. Diagnosed on
    # job 723ea5d2 (colmap_images): removed 0.84% of the model (all visible
    # scattered specks around the scan) while leaving a real, coherent ~19%
    # chunk of the model untouched - that chunk came back as its own
    # connected component purely from a local density gap, not because it's
    # debris (confirmed: it sat 100% inside the main component's bounding
    # box, which is exactly what this filter checks for before removing
    # anything).
    cluster_filter_enabled: bool = True
    cluster_filter_radius_multiplier: float = 5.0

    # Third, separate post-training filter pass (see
    # floater_filter.filter_photometric_inconsistency): removes splats whose
    # live-training photometric residual (photo_resid_avg, only present when
    # photo_consistency_enabled was set for the job) sits in the extreme tail
    # of this model's own distribution. Built after position_anchor/aniso_reg
    # (training-time) and the two filters above (geometric/shape-based) all
    # failed to isolate the AU_Las_Vegas floating-tile artifact - see that
    # investigation report. Only takes effect when the model actually has
    # photo_resid_avg data (no-ops otherwise).
    # DEFAULT OFF: measured against the real training photos (59 views,
    # PSNR of render vs photo) it makes the model WORSE - 18.15 dB unfiltered
    # -> 17.53 dB at percentile 97.5 -> 17.24 dB at 95. The "floating tile"
    # artifact it was built to remove turned out to be real floor inlay (the
    # model reproduces it tile for tile vs the real photos); the splats it
    # removes are mostly large blurry foreground/haze splats that still carry
    # real image content. Kept as an opt-in diagnostic only.
    photo_consistency_filter_enabled: bool = False
    photo_consistency_filter_percentile: float = 97.5
    photo_consistency_filter_min_observations: int = 30


@dataclass
class MaskingConfig:
    """DeepLabV3 segmentation masking settings"""
    enabled: bool = False  # По умолчанию отключено; включать точечно для динамичных сцен
    remove_classes: list = field(default_factory=lambda: ["bicycle", "bus", "car", "cat", "dog", "motorbike", "person"])
    
    
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
    if brush_dir := os.getenv("GS_BRUSH_DIR"):
        config.paths.brush_dir = Path(brush_dir)
    if brush_exe := os.getenv("GS_BRUSH_EXE"):
        config.paths.brush_exe = Path(brush_exe)
        
    # COLMAP
    config.colmap.use_gpu = os.getenv("COLMAP_USE_GPU", "true").lower() == "true"
    config.colmap.camera_model = os.getenv("COLMAP_CAMERA_MODEL", config.colmap.camera_model)
    config.colmap.single_camera = os.getenv("COLMAP_SINGLE_CAMERA", str(config.colmap.single_camera).lower()).lower() == "true"
    config.colmap.max_image_size = int(os.getenv("COLMAP_MAX_IMAGE_SIZE", config.colmap.max_image_size))
    config.colmap.sift_max_features = int(os.getenv("COLMAP_SIFT_MAX_FEATURES", config.colmap.sift_max_features))
    config.colmap.matcher_type = os.getenv("COLMAP_MATCHER_TYPE", config.colmap.matcher_type)
    config.colmap.sequential_loop_detection = os.getenv(
        "COLMAP_SEQUENTIAL_LOOP_DETECTION", str(config.colmap.sequential_loop_detection).lower()
    ).lower() == "true"
    if vocab_tree := os.getenv("COLMAP_VOCAB_TREE_PATH"):
        config.colmap.vocab_tree_path = Path(vocab_tree)
    
    # Brush  
    config.brush.total_steps = int(os.getenv("BRUSH_TOTAL_STEPS", config.brush.total_steps))
    config.brush.max_resolution = int(os.getenv("BRUSH_MAX_RESOLUTION", config.brush.max_resolution))
    config.brush.eval_every = int(os.getenv("BRUSH_EVAL_EVERY", config.brush.eval_every))
    config.brush.export_every = int(os.getenv("BRUSH_EXPORT_EVERY", config.brush.export_every))
    config.brush.floater_filter_enabled = os.getenv(
        "BRUSH_FLOATER_FILTER_ENABLED", str(config.brush.floater_filter_enabled).lower()
    ).lower() == "true"
    config.brush.floater_filter_aniso_threshold = float(
        os.getenv("BRUSH_FLOATER_FILTER_ANISO_THRESHOLD", config.brush.floater_filter_aniso_threshold)
    )
    config.brush.floater_filter_size_multiplier = float(
        os.getenv("BRUSH_FLOATER_FILTER_SIZE_MULTIPLIER", config.brush.floater_filter_size_multiplier)
    )
    config.brush.cluster_filter_enabled = os.getenv(
        "BRUSH_CLUSTER_FILTER_ENABLED", str(config.brush.cluster_filter_enabled).lower()
    ).lower() == "true"
    config.brush.cluster_filter_radius_multiplier = float(
        os.getenv("BRUSH_CLUSTER_FILTER_RADIUS_MULTIPLIER", config.brush.cluster_filter_radius_multiplier)
    )
    config.brush.aniso_reg_weight = float(
        os.getenv("BRUSH_ANISO_REG_WEIGHT", config.brush.aniso_reg_weight)
    )
    config.brush.aniso_reg_ratio_threshold = float(
        os.getenv("BRUSH_ANISO_REG_RATIO_THRESHOLD", config.brush.aniso_reg_ratio_threshold)
    )
    config.brush.depth_loss_weight = float(
        os.getenv("BRUSH_DEPTH_LOSS_WEIGHT", config.brush.depth_loss_weight)
    )
    config.brush.depth_max_range = float(
        os.getenv("BRUSH_DEPTH_MAX_RANGE", config.brush.depth_max_range)
    )
    config.brush.position_anchor_weight = float(
        os.getenv("BRUSH_POSITION_ANCHOR_WEIGHT", config.brush.position_anchor_weight)
    )
    config.brush.position_anchor_max_dist = float(
        os.getenv("BRUSH_POSITION_ANCHOR_MAX_DIST", config.brush.position_anchor_max_dist)
    )
    config.brush.photo_consistency_enabled = os.getenv(
        "BRUSH_PHOTO_CONSISTENCY_ENABLED", str(config.brush.photo_consistency_enabled).lower()
    ).lower() == "true"
    config.brush.photo_consistency_warmup_steps = int(
        os.getenv("BRUSH_PHOTO_CONSISTENCY_WARMUP_STEPS", config.brush.photo_consistency_warmup_steps)
    )
    config.brush.photo_consistency_filter_enabled = os.getenv(
        "BRUSH_PHOTO_CONSISTENCY_FILTER_ENABLED", str(config.brush.photo_consistency_filter_enabled).lower()
    ).lower() == "true"
    config.brush.photo_consistency_filter_percentile = float(
        os.getenv("BRUSH_PHOTO_CONSISTENCY_FILTER_PERCENTILE", config.brush.photo_consistency_filter_percentile)
    )
    config.brush.photo_consistency_filter_min_observations = int(
        os.getenv(
            "BRUSH_PHOTO_CONSISTENCY_FILTER_MIN_OBSERVATIONS",
            config.brush.photo_consistency_filter_min_observations,
        )
    )

    # Masking
    config.masking.enabled = os.getenv("MASKING_ENABLED", str(config.masking.enabled).lower()).lower() == "true"
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
