"""
Job Manager - queue management and job state handling
"""

import asyncio
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
import aiofiles
import aiofiles.os

from .models import (
    JobStatus, JobInfo, JobProgress, ColmapProgress,
    TrainingProgress, ColmapStage, WebSocketMessage, TrainingMode, PreviewStatus
)
from .config import get_config


@dataclass
class Job:
    """Internal job representation"""
    job_id: str
    name: str
    status: JobStatus
    progress: JobProgress
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    images_count: int = 0
    total_size_bytes: int = 0
    
    model_path: Optional[str] = None
    model_size_bytes: Optional[int] = None

    # Видео-превью - см. preview_renderer.py. Никогда не выставляется
    # автоматически, только через явный POST /jobs/{id}/preview.
    preview_status: PreviewStatus = PreviewStatus.NOT_REQUESTED
    preview_error: Optional[str] = None
    preview_stats: Optional[Dict[str, Any]] = None

    config: Dict[str, Any] = field(default_factory=dict)

    # Внутренние пути
    job_dir: Optional[Path] = None
    images_dir: Optional[Path] = None
    colmap_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Callbacks
    _websocket_callbacks: List[Callable] = field(default_factory=list)

    @property
    def preview_path(self) -> Optional[Path]:
        return self.output_dir / "preview.mp4" if self.output_dir else None

    def to_info(self) -> JobInfo:
        """Конвертировать в JobInfo для API"""
        return JobInfo(
            job_id=self.job_id,
            name=self.name,
            status=self.status,
            progress=self.progress,
            created_at=self.created_at,
            updated_at=self.updated_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            images_count=self.images_count,
            total_size_bytes=self.total_size_bytes,
            model_path=self.model_path,
            model_size_bytes=self.model_size_bytes,
            preview_status=self.preview_status,
            preview_error=self.preview_error,
            config=self.config
        )

    def to_dict(self) -> dict:
        """Сериализовать для сохранения"""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress.model_dump(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "images_count": self.images_count,
            "total_size_bytes": self.total_size_bytes,
            "model_path": self.model_path,
            "model_size_bytes": self.model_size_bytes,
            "preview_status": self.preview_status.value,
            "preview_error": self.preview_error,
            "preview_stats": self.preview_stats,
            "config": self.config
        }
    
    @classmethod
    def from_dict(cls, data: dict, job_dir: Path) -> "Job":
        """Десериализовать из сохранения"""
        progress_data = data.get("progress", {})
        
        # Восстановить вложенные объекты прогресса
        colmap_data = progress_data.get("colmap")
        training_data = progress_data.get("training")
        
        progress = JobProgress(
            overall_progress=progress_data.get("overall_progress", 0),
            status=JobStatus(progress_data.get("status", "pending")),
            colmap=ColmapProgress(**colmap_data) if colmap_data else None,
            training=TrainingProgress(**training_data) if training_data else None,
            message=progress_data.get("message", ""),
            error=progress_data.get("error")
        )
        
        job = cls(
            job_id=data["job_id"],
            name=data["name"],
            status=JobStatus(data["status"]),
            progress=progress,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            images_count=data.get("images_count", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            model_path=data.get("model_path"),
            model_size_bytes=data.get("model_size_bytes"),
            preview_status=PreviewStatus(data.get("preview_status", "not_requested")),
            preview_error=data.get("preview_error"),
            preview_stats=data.get("preview_stats"),
            config=data.get("config", {})
        )
        
        job.job_dir = job_dir
        job.images_dir = job_dir / "images"
        job.colmap_dir = job_dir / "colmap"
        job.output_dir = job_dir / "output"
        
        return job


class JobManager:
    """Manager for all jobs"""
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._processing_lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._websocket_subscribers: Dict[str, List[Callable]] = {}
        self._config = get_config()
        
    async def initialize(self):
        """Initialize manager, load saved jobs"""
        jobs_dir = self._config.paths.jobs_dir
        
        if jobs_dir.exists():
            for job_dir in jobs_dir.iterdir():
                if job_dir.is_dir():
                    state_file = job_dir / "state.json"
                    if state_file.exists():
                        try:
                            async with aiofiles.open(state_file, 'r', encoding='utf-8') as f:
                                data = json.loads(await f.read())
                                job = Job.from_dict(data, job_dir)
                                self._jobs[job.job_id] = job
                                
                                # Restart incomplete jobs
                                if job.status in [JobStatus.COLMAP_RUNNING, JobStatus.TRAINING]:
                                    job.status = JobStatus.PENDING
                                    job.progress.status = JobStatus.PENDING
                                    job.progress.message = "Restart after server restart"
                                    await self._queue.put(job.job_id)

                                # A preview render doesn't survive a restart (not queued,
                                # not resumed automatically - the client has to re-request it).
                                if job.preview_status == PreviewStatus.RENDERING:
                                    job.preview_status = PreviewStatus.FAILED
                                    job.preview_error = "Interrupted by server restart"


                        except Exception as e:
                            print(f"Error loading job {job_dir}: {e}")
        
        # Запустить воркер
        self._worker_task = asyncio.create_task(self._process_queue())
        
    async def shutdown(self):
        """Graceful shutdown"""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
                
        # Сохранить все состояния
        for job in self._jobs.values():
            await self._save_job_state(job)
            
    async def create_job(
        self, 
        name: str, 
        colmap_config: Optional[dict] = None,
        brush_config: Optional[dict] = None,
        training_mode: TrainingMode = TrainingMode.COLMAP_IMAGES,
        pointcloud_path: Optional[str] = None,
        skip_colmap: bool = False,
        dataset_path: Optional[str] = None,
        auto_start: bool = True,
        hybrid_config: Optional[dict] = None
    ) -> Job:
        """Create a new job"""
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        job = Job(
            job_id=job_id,
            name=name,
            status=JobStatus.PENDING,
            progress=JobProgress(status=JobStatus.PENDING, message="Waiting for file upload"),
            created_at=now,
            updated_at=now,
            config={
                "colmap": colmap_config or {},
                "brush": brush_config or {},
                "hybrid_known_poses": hybrid_config or {},
                "training_mode": training_mode.value,
                "pointcloud_path": pointcloud_path,
                "skip_colmap": skip_colmap,
                "dataset_path": dataset_path,
                "auto_start": auto_start
            }
        )
        
        # Создать директории
        job.job_dir = self._config.paths.jobs_dir / job_id
        job.images_dir = job.job_dir / "images"
        job.colmap_dir = job.job_dir / "colmap"
        job.output_dir = job.job_dir / "output"
        
        job.job_dir.mkdir(parents=True, exist_ok=True)
        job.images_dir.mkdir(exist_ok=True)
        job.colmap_dir.mkdir(exist_ok=True)
        job.output_dir.mkdir(exist_ok=True)
        
        self._jobs[job_id] = job
        await self._save_job_state(job)
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    async def list_jobs(
        self, 
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Job]:
        """Get list of jobs"""
        jobs = list(self._jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
            
        # Sort by creation date (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs[offset:offset + limit]
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        job = self._jobs.get(job_id)
        if not job:
            return False
            
        # Cannot delete a running job
        if job.status in [JobStatus.COLMAP_RUNNING, JobStatus.TRAINING, JobStatus.UPLOADING]:
            return False
            
        # Удалить файлы
        if job.job_dir and job.job_dir.exists():
            shutil.rmtree(job.job_dir)
            
        del self._jobs[job_id]
        return True
    
    async def start_processing(self, job_id: str) -> bool:
        """Start job processing"""
        job = self._jobs.get(job_id)
        if not job:
            return False
            
        if job.status not in [JobStatus.UPLOADED, JobStatus.COLMAP_DONE, JobStatus.PENDING]:
            return False
            
        await self._queue.put(job_id)
        return True
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self._jobs.get(job_id)
        if not job:
            return False
            
        job.status = JobStatus.CANCELLED
        job.progress.status = JobStatus.CANCELLED
        job.progress.message = "Cancelled by user"
        job.updated_at = datetime.utcnow()
        
        await self._save_job_state(job)
        await self._notify_subscribers(job, "cancelled", {"message": "Job cancelled"})
        
        return True
    
    async def reset_job_for_restart(self, job_id: str) -> bool:
        """Reset job status for restart"""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        job.status = JobStatus.UPLOADED
        job.progress = JobProgress(
            overall_progress=0.0,
            status=JobStatus.UPLOADED,
            message="Ready for restart"
        )
        job.updated_at = datetime.utcnow()
        job.started_at = None
        job.completed_at = None
        job.model_path = None
        job.model_size_bytes = None
        
        await self._save_job_state(job)
        return True
    
    async def update_job_progress(
        self, 
        job_id: str, 
        status: Optional[JobStatus] = None,
        message: Optional[str] = None,
        colmap_progress: Optional[ColmapProgress] = None,
        training_progress: Optional[TrainingProgress] = None,
        overall_progress: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Update job progress"""
        job = self._jobs.get(job_id)
        if not job:
            return
            
        if status:
            job.status = status
            job.progress.status = status
            
        if message:
            job.progress.message = message
            
        if colmap_progress:
            job.progress.colmap = colmap_progress
            
        if training_progress:
            job.progress.training = training_progress
            
        if overall_progress is not None:
            job.progress.overall_progress = overall_progress
            
        if error:
            job.progress.error = error
            
        job.updated_at = datetime.utcnow()
        
        # Уведомить подписчиков
        await self._notify_subscribers(job, "progress", job.progress.model_dump())
        
        # Периодически сохранять состояние
        await self._save_job_state(job)
    
    async def mark_upload_complete(self, job_id: str, images_count: int, total_size: int):
        """Mark upload as complete"""
        job = self._jobs.get(job_id)
        if not job:
            return
            
        job.images_count = images_count
        job.total_size_bytes = total_size
        job.status = JobStatus.UPLOADED
        job.progress.status = JobStatus.UPLOADED
        job.progress.message = f"Uploaded {images_count} images"
        job.progress.overall_progress = 5.0
        job.updated_at = datetime.utcnow()
        
        await self._save_job_state(job)
        await self._notify_subscribers(job, "upload_complete", {
            "images_count": images_count,
            "total_size": total_size
        })
        
        # Автозапуск если включен
        if job.config.get("auto_start", True):
            await self._queue.put(job_id)
    
    async def mark_completed(self, job_id: str, model_path: str, model_size: int):
        """Mark processing as completed"""
        job = self._jobs.get(job_id)
        if not job:
            return
            
        job.status = JobStatus.COMPLETED
        job.progress.status = JobStatus.COMPLETED
        job.progress.message = "Training completed"
        job.progress.overall_progress = 100.0
        job.model_path = model_path
        job.model_size_bytes = model_size
        job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
        
        await self._save_job_state(job)
        await self._notify_subscribers(job, "completed", {
            "model_path": model_path,
            "model_size": model_size
        })
    
    async def mark_failed(self, job_id: str, error: str):
        """Mark job as failed"""
        job = self._jobs.get(job_id)
        if not job:
            return
            
        job.status = JobStatus.FAILED
        job.progress.status = JobStatus.FAILED
        job.progress.error = error
        job.progress.message = f"Error: {error}"
        job.updated_at = datetime.utcnow()
        
        await self._save_job_state(job)
        await self._notify_subscribers(job, "error", {"error": error})
    
    async def start_preview_render(self, job_id: str) -> bool:
        """Mark a preview render as started. Returns False if one is already
        in progress for this job (caller should treat that as a conflict)."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.preview_status == PreviewStatus.RENDERING:
            return False

        job.preview_status = PreviewStatus.RENDERING
        job.preview_error = None
        job.preview_stats = None
        job.updated_at = datetime.utcnow()
        await self._save_job_state(job)
        return True

    async def mark_preview_completed(self, job_id: str, stats: dict):
        job = self._jobs.get(job_id)
        if not job:
            return
        job.preview_status = PreviewStatus.COMPLETED
        job.preview_error = None
        job.preview_stats = stats
        job.updated_at = datetime.utcnow()
        await self._save_job_state(job)
        await self._notify_subscribers(job, "preview_completed", {"stats": stats})

    async def mark_preview_failed(self, job_id: str, error: str):
        job = self._jobs.get(job_id)
        if not job:
            return
        job.preview_status = PreviewStatus.FAILED
        job.preview_error = error
        job.updated_at = datetime.utcnow()
        await self._save_job_state(job)
        await self._notify_subscribers(job, "preview_failed", {"error": error})

    def subscribe(self, job_id: str, callback: Callable):
        """Subscribe to job updates"""
        if job_id not in self._websocket_subscribers:
            self._websocket_subscribers[job_id] = []
        self._websocket_subscribers[job_id].append(callback)
        
    def unsubscribe(self, job_id: str, callback: Callable):
        """Unsubscribe from updates"""
        if job_id in self._websocket_subscribers:
            try:
                self._websocket_subscribers[job_id].remove(callback)
            except ValueError:
                pass
    
    async def _notify_subscribers(self, job: Job, msg_type: str, data: dict):
        """Notify all subscribers"""
        callbacks = self._websocket_subscribers.get(job.job_id, [])
        
        message = WebSocketMessage(
            type=msg_type,
            job_id=job.job_id,
            data=data,
            timestamp=datetime.utcnow()
        )
        
        for callback in callbacks:
            try:
                await callback(message)
            except Exception as e:
                print(f"Error sending WebSocket: {e}")
    
    async def _save_job_state(self, job: Job):
        """Save job state to disk"""
        if not job.job_dir:
            return
            
        state_file = job.job_dir / "state.json"
        async with aiofiles.open(state_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(job.to_dict(), indent=2, ensure_ascii=False))
    
    async def _process_queue(self):
        """Queue processing worker"""
        from .colmap_runner import ColmapRunner
        from .brush_runner import BrushRunner
        from .mask_generator import MaskGenerator
        from .known_pose_triangulation import KnownPoseTriangulator
        from .utils import normalize_exif_orientation

        colmap = ColmapRunner(self)
        brush = BrushRunner(self)
        mask_gen = MaskGenerator(self)
        known_pose_triangulator = KnownPoseTriangulator(self)
        masking_enabled = get_config().masking.enabled
        
        while True:
            try:
                job_id = await self._queue.get()
                job = self._jobs.get(job_id)
                
                if not job or job.status == JobStatus.CANCELLED:
                    continue
                
                async with self._processing_lock:
                    try:
                        job.started_at = datetime.utcnow()

                        # Resolve mode with backward compatibility.
                        mode_raw = job.config.get("training_mode")
                        if mode_raw:
                            training_mode = TrainingMode(mode_raw)
                        else:
                            training_mode = (
                                TrainingMode.POINTCLOUD_ONLY
                                if bool(job.config.get("skip_colmap", False))
                                else TrainingMode.COLMAP_IMAGES
                            )

                        skip_colmap = training_mode in (
                            TrainingMode.POINTCLOUD_ONLY, TrainingMode.HYBRID_KNOWN_POSES
                        )
                        prepared_dataset_path = job.config.get("dataset_path")
                        pointcloud_path = job.config.get("pointcloud_path")

                        if training_mode == TrainingMode.POINTCLOUD_ONLY and not prepared_dataset_path:
                            await self.mark_failed(job_id, "pointcloud_only mode requires dataset_path")
                            continue

                        if training_mode == TrainingMode.HYBRID_KNOWN_POSES and not prepared_dataset_path:
                            await self.mark_failed(job_id, "hybrid_known_poses mode requires dataset_path")
                            continue

                        if training_mode == TrainingMode.HYBRID_COLMAP_POINTCLOUD and not pointcloud_path:
                            await self.mark_failed(job_id, "hybrid_colmap_pointcloud mode requires pointcloud_path")
                            continue

                        # Stage 0a: Normalize EXIF orientation in uploaded images
                        if job.status in [JobStatus.UPLOADED, JobStatus.PENDING] and not skip_colmap:
                            await self.update_job_progress(
                                job_id,
                                message="Normalizing image orientations...",
                                overall_progress=2.0,
                            )
                            loop = asyncio.get_event_loop()
                            fixed = await loop.run_in_executor(
                                None, normalize_exif_orientation, job.images_dir
                            )

                        # Stage 0b: Generate masks (if enabled)
                        if job.status in [JobStatus.UPLOADED, JobStatus.PENDING] and not skip_colmap:
                            job_mask_enabled = job.config.get("masking", {}).get("enabled", masking_enabled)
                            if job_mask_enabled:
                                masks_dir = await mask_gen.run(job)
                                job.config["_masks_dir"] = str(masks_dir)

                        # Stage 1: COLMAP (if needed)
                        if job.status in [JobStatus.UPLOADED, JobStatus.PENDING]:
                            if training_mode == TrainingMode.HYBRID_KNOWN_POSES:
                                ok = await known_pose_triangulator.run(job, Path(prepared_dataset_path))
                                if not ok:
                                    continue
                            elif skip_colmap:
                                await self.update_job_progress(
                                    job_id,
                                    status=JobStatus.COLMAP_DONE,
                                    message="Skipping COLMAP: using prepared dataset",
                                    overall_progress=50.0,
                                )
                            else:
                                await colmap.run(job)

                                if job.status == JobStatus.COLMAP_FAILED:
                                    continue

                                if training_mode == TrainingMode.HYBRID_COLMAP_POINTCLOUD:
                                    source_path = Path(pointcloud_path)
                                    if not source_path.exists():
                                        await self.mark_failed(job_id, f"Point cloud file not found: {source_path}")
                                        continue

                                    dense_dir = job.colmap_dir / "dense"
                                    dense_dir.mkdir(parents=True, exist_ok=True)
                                    dest_path = dense_dir / "points3d.ply"
                                    shutil.copy2(source_path, dest_path)

                                    await self.update_job_progress(
                                        job_id,
                                        message=f"Hybrid mode: external point cloud injected ({source_path.name})",
                                        overall_progress=50.0,
                                    )
                        
                        # Stage 2: Brush training
                        if job.status == JobStatus.COLMAP_DONE:
                            await brush.run(job)
                            
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        await self.mark_failed(job_id, str(e))
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker error: {e}")
                await asyncio.sleep(1)


# Глобальный экземпляр
_job_manager: Optional[JobManager] = None

def get_job_manager() -> JobManager:
    """Get global job manager"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
