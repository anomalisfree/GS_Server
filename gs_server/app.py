"""
GS Server - FastAPI server for automatic Gaussian Splatting training
"""

import os
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect, Body, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from .config import get_config, AppConfig
from .models import (
    JobStatus, JobInfo, JobProgress, CreateJobRequest, CreateJobResponse,
    JobListResponse, UploadProgressResponse, StartTrainingRequest,
    ModelInfo, ServerStatus, WebSocketMessage, TrainingMode,
    PreviewRequest, PreviewInfo, PreviewStatus
)
from .job_manager import get_job_manager, JobManager
from . import __version__


# Lifespan для инициализации/завершения
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Initialization
    config = get_config()
    job_manager = get_job_manager()
    await job_manager.initialize()
    
    print(f"GS Server v{__version__} started")
    print(f"  Jobs dir: {config.paths.jobs_dir}")
    print(f"  COLMAP: {config.paths.colmap_exe}")
    print(f"  Brush: {config.paths.brush_dir}")
    
    yield
    
    # Shutdown
    await job_manager.shutdown()
    print("GS Server stopped")


# Создание приложения
app = FastAPI(
    title="Gaussian Splatting Training Server",
    description="Server for automatic image processing via COLMAP and Gaussian Splatting training with Brush",
    version=__version__,
    lifespan=lifespan
)

# Сериализует on-demand рендеры превью между собой (не с обучением — то и
# другое использует GPU, но параллельные превью-рендеры быстрее исчерпают
# память, чем один превью параллельно с трейном).
_preview_render_lock = asyncio.Lock()

def _public_base_url(request: Request) -> str:
    """Build an externally-reachable base URL from the actual request the
    client made, instead of config.server.host (which is the bind address,
    typically 0.0.0.0 - not reachable as a URL by clients)."""
    return str(request.base_url).rstrip("/")


# CORS для доступа из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_training_mode(
    explicit_mode: Optional[TrainingMode],
    skip_colmap: bool,
) -> TrainingMode:
    """Resolve training mode with backward compatibility for skip_colmap."""
    if explicit_mode is not None:
        return explicit_mode
    if skip_colmap:
        return TrainingMode.POINTCLOUD_ONLY
    return TrainingMode.COLMAP_IMAGES


# ============== API Endpoints ==============

@app.get("/", tags=["Info"])
async def root():
    """Главная страница"""
    return {
        "name": "GS Training Server",
        "version": __version__,
        "docs": "/docs"
    }


@app.get("/status", response_model=ServerStatus, tags=["Info"])
async def get_server_status():
    """Получить статус сервера"""
    config = get_config()
    job_manager = get_job_manager()
    
    jobs = await job_manager.list_jobs(limit=1000)
    
    active = sum(1 for j in jobs if j.status in [JobStatus.COLMAP_RUNNING, JobStatus.TRAINING])
    queued = sum(1 for j in jobs if j.status in [JobStatus.PENDING, JobStatus.UPLOADED])
    completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
    
    # Проверить GPU
    gpu_available = config.colmap.use_gpu  # Упрощенно
    
    return ServerStatus(
        status="running",
        version=__version__,
        active_jobs=active,
        queued_jobs=queued,
        completed_jobs=completed,
        gpu_available=gpu_available
    )


# ============== Jobs API ==============

@app.post("/jobs", response_model=CreateJobResponse, tags=["Jobs"])
async def create_job(
    http_request: Request,
    request: CreateJobRequest = Body(
        ...,
        description=(
            "Create a training job and choose one of the supported training modes: "
            "colmap_images, pointcloud_only, hybrid_colmap_pointcloud."
        )
    )
):
    """Создать новую задачу с явным выбором training mode"""
    job_manager = get_job_manager()

    training_mode = _resolve_training_mode(request.training_mode, request.skip_colmap)

    job = await job_manager.create_job(
        name=request.name,
        colmap_config=request.colmap_config,
        brush_config=request.brush_config,
        training_mode=training_mode,
        pointcloud_path=request.pointcloud_path,
        skip_colmap=request.skip_colmap,
        dataset_path=request.dataset_path,
        auto_start=request.auto_start
    )

    if (
        request.auto_start
        and training_mode in (TrainingMode.POINTCLOUD_ONLY, TrainingMode.HYBRID_KNOWN_POSES)
        and request.dataset_path
    ):
        await job_manager.start_processing(job.job_id)

    base_url = _public_base_url(http_request)
    ws_url = "ws" + base_url.split("http", 1)[1]  # http(s):// -> ws(s)://

    return CreateJobResponse(
        job_id=job.job_id,
        upload_url=f"{base_url}/jobs/{job.job_id}/upload",
        websocket_url=f"{ws_url}/jobs/{job.job_id}/ws"
    )


@app.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Получить список задач"""
    job_manager = get_job_manager()
    jobs = await job_manager.list_jobs(status=status, limit=limit, offset=offset)
    
    return JobListResponse(
        jobs=[j.to_info() for j in jobs],
        total=len(jobs)
    )


@app.get("/jobs/{job_id}", response_model=JobInfo, tags=["Jobs"])
async def get_job(job_id: str):
    """Получить информацию о задаче"""
    job_manager = get_job_manager()
    job = await job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.to_info()


@app.get("/jobs/{job_id}/progress", response_model=JobProgress, tags=["Jobs"])
async def get_job_progress(job_id: str):
    """Получить прогресс задачи"""
    job_manager = get_job_manager()
    job = await job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.progress


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """Удалить задачу"""
    job_manager = get_job_manager()
    
    success = await job_manager.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete job (not found or in progress)")
    
    return {"status": "deleted", "job_id": job_id}


@app.post("/jobs/{job_id}/cancel", tags=["Jobs"])
async def cancel_job(job_id: str):
    """Отменить задачу"""
    job_manager = get_job_manager()
    
    success = await job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job")
    
    return {"status": "cancelled", "job_id": job_id}


@app.post("/jobs/{job_id}/restart", tags=["Jobs"])
async def restart_job(job_id: str):
    """Перезапустить неудачную задачу"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Можно перезапустить только неудачные задачи
    failed_statuses = {JobStatus.COLMAP_FAILED, JobStatus.TRAINING_FAILED, JobStatus.FAILED, JobStatus.CANCELLED}
    if job.status not in failed_statuses:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot restart job with status '{job.status.value}'. Only failed/cancelled jobs can be restarted."
        )
    
    # Очистить папки colmap и output
    colmap_dir = job.colmap_dir
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)
        colmap_dir.mkdir()
    
    output_dir = job.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir()
    
    # Сбросить статус и запустить
    await job_manager.reset_job_for_restart(job_id)
    success = await job_manager.start_processing(job_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to restart job")
    
    return {"status": "restarted", "job_id": job_id}


@app.post("/jobs/{job_id}/start", tags=["Jobs"])
async def start_job(
    job_id: str,
    request: Optional[StartTrainingRequest] = Body(
        default=None,
        description=(
            "Optionally override mode/paths/brush settings before manual start. "
            "Useful for pointcloud_only and hybrid_colmap_pointcloud flows."
        )
    )
):
    """Запустить обработку задачи вручную (с optional overrides)"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Обновить конфиг если передан
    if request:
        if request.training_mode is not None:
            job.config["training_mode"] = request.training_mode.value
            job.config["skip_colmap"] = request.training_mode == TrainingMode.POINTCLOUD_ONLY
        if request.brush_config:
            job.config["brush"] = {**job.config.get("brush", {}), **request.brush_config}
        if request.skip_colmap:
            job.config["skip_colmap"] = True
            job.config["training_mode"] = TrainingMode.POINTCLOUD_ONLY.value
        if request.dataset_path:
            job.config["dataset_path"] = request.dataset_path
        if request.pointcloud_path:
            job.config["pointcloud_path"] = request.pointcloud_path
    
    success = await job_manager.start_processing(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot start job (invalid status)")
    
    return {"status": "started", "job_id": job_id}


# ============== Upload API ==============

@app.post("/jobs/{job_id}/upload", tags=["Upload"])
async def upload_images(
    job_id: str,
    files: List[UploadFile] = File(...)
):
    """Загрузить изображения для задачи"""
    job_manager = get_job_manager()
    config = get_config()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in [JobStatus.PENDING, JobStatus.UPLOADING]:
        raise HTTPException(status_code=400, detail="Job is not accepting uploads")
    
    # Update status
    await job_manager.update_job_progress(
        job_id,
        status=JobStatus.UPLOADING,
        message=f"Uploading {len(files)} files..."
    )
    
    images_dir = job.images_dir
    total_size = 0
    uploaded_count = 0
    
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    for file in files:
        # Проверить расширение
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            continue
        
        # Безопасное имя файла
        safe_name = Path(file.filename).name
        file_path = images_dir / safe_name
        
        # Сохранить файл
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
                total_size += len(chunk)
        
        uploaded_count += 1
    
    # Подсчитать все файлы в директории
    all_images = list(images_dir.glob('*'))
    total_images = len([f for f in all_images if f.suffix.lower() in allowed_extensions])
    total_size = sum(f.stat().st_size for f in all_images if f.is_file())
    
    return {
        "uploaded": uploaded_count,
        "total_images": total_images,
        "total_size_bytes": total_size
    }


@app.post("/jobs/{job_id}/pointcloud", tags=["Upload"])
async def upload_pointcloud(
    job_id: str,
    file: UploadFile = File(
        ...,
        description="External points3d.ply file for hybrid_colmap_pointcloud mode"
    )
):
    """Upload external points3d.ply for hybrid mode."""
    job_manager = get_job_manager()

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    filename = Path(file.filename).name
    if Path(filename).suffix.lower() != ".ply":
        raise HTTPException(status_code=400, detail="Only .ply point cloud is supported")

    inputs_dir = job.job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    final_path = inputs_dir / "points3d.ply"

    total_size = 0
    async with aiofiles.open(final_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)
            total_size += len(chunk)

    job.config["pointcloud_path"] = str(final_path)
    await job_manager.update_job_progress(
        job_id,
        message=f"Point cloud uploaded: {filename} ({total_size} bytes)"
    )

    return {
        "status": "pointcloud_uploaded",
        "filename": filename,
        "saved_as": str(final_path),
        "size_bytes": total_size
    }


@app.post("/jobs/{job_id}/pointcloud/chunk", tags=["Upload"])
async def upload_pointcloud_chunk(
    job_id: str,
    filename: str = Form(..., description="Original point cloud filename, must have .ply extension"),
    chunk_index: int = Form(..., description="Zero-based chunk index"),
    total_chunks: int = Form(..., description="Total number of chunks in this upload"),
    chunk: UploadFile = File(..., description="Binary payload for this chunk")
):
    """Upload external point cloud in chunks for hybrid mode."""
    job_manager = get_job_manager()

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if Path(filename).suffix.lower() != ".ply":
        raise HTTPException(status_code=400, detail="Only .ply point cloud is supported")

    chunks_dir = job.job_dir / "chunks" / "pointcloud"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = chunks_dir / f"{chunk_index:06d}"
    async with aiofiles.open(chunk_path, 'wb') as f:
        await f.write(await chunk.read())

    chunks = list(chunks_dir.glob('*'))
    if len(chunks) == total_chunks:
        inputs_dir = job.job_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        final_path = inputs_dir / "points3d.ply"

        async with aiofiles.open(final_path, 'wb') as f:
            for i in range(total_chunks):
                chunk_file = chunks_dir / f"{i:06d}"
                async with aiofiles.open(chunk_file, 'rb') as cf:
                    await f.write(await cf.read())

        shutil.rmtree(chunks_dir)

        job.config["pointcloud_path"] = str(final_path)
        await job_manager.update_job_progress(
            job_id,
            message=f"Point cloud uploaded in chunks: {filename}"
        )

        return {
            "status": "complete",
            "filename": filename,
            "saved_as": str(final_path)
        }

    return {
        "status": "chunk_received",
        "chunk_index": chunk_index,
        "received_chunks": len(chunks),
        "total_chunks": total_chunks
    }


@app.post("/jobs/{job_id}/upload/complete", tags=["Upload"])
async def complete_upload(job_id: str):
    """Завершить загрузку и начать обработку"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Подсчитать файлы
    images_dir = job.images_dir
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    images = [f for f in images_dir.glob('*') if f.suffix.lower() in allowed_extensions]
    images_count = len(images)
    total_size = sum(f.stat().st_size for f in images)
    
    if images_count == 0:
        raise HTTPException(status_code=400, detail="No images uploaded")
    
    await job_manager.mark_upload_complete(job_id, images_count, total_size)
    
    return {
        "status": "upload_complete",
        "images_count": images_count,
        "total_size_bytes": total_size
    }


@app.post("/jobs/{job_id}/upload/chunk", tags=["Upload"])
async def upload_chunk(
    job_id: str,
    filename: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    chunk: UploadFile = File(...)
):
    """Загрузить часть большого файла (chunked upload)"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Временная директория для chunks
    chunks_dir = job.job_dir / "chunks" / filename
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохранить chunk
    chunk_path = chunks_dir / f"{chunk_index:06d}"
    async with aiofiles.open(chunk_path, 'wb') as f:
        await f.write(await chunk.read())
    
    # Проверить, все ли chunks загружены
    chunks = list(chunks_dir.glob('*'))
    if len(chunks) == total_chunks:
        # Собрать файл
        final_path = job.images_dir / filename
        async with aiofiles.open(final_path, 'wb') as f:
            for i in range(total_chunks):
                chunk_file = chunks_dir / f"{i:06d}"
                async with aiofiles.open(chunk_file, 'rb') as cf:
                    await f.write(await cf.read())
        
        # Удалить chunks
        shutil.rmtree(chunks_dir)
        
        return {"status": "complete", "filename": filename}
    
    return {
        "status": "chunk_received",
        "chunk_index": chunk_index,
        "received_chunks": len(chunks),
        "total_chunks": total_chunks
    }


# ============== Download API ==============

@app.get("/jobs/{job_id}/model", tags=["Download"])
async def download_model(job_id: str):
    """Скачать обученную модель"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed")
    
    if not job.model_path or not Path(job.model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        job.model_path,
        filename=f"{job.name}_model.ply",
        media_type="application/octet-stream"
    )


@app.get("/jobs/{job_id}/model/info", response_model=ModelInfo, tags=["Download"])
async def get_model_info(job_id: str, http_request: Request):
    """Получить информацию о модели"""
    job_manager = get_job_manager()

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed")
    
    training = job.progress.training
    
    return ModelInfo(
        job_id=job_id,
        model_path=job.model_path,
        model_size_bytes=job.model_size_bytes or 0,
        format="ply",
        training_steps=training.total_steps if training else 0,
        final_psnr=training.psnr if training else None,
        final_ssim=training.ssim if training else None,
        splat_count=training.splat_count if training else 0,
        created_at=job.completed_at or job.updated_at,
        download_url=f"{_public_base_url(http_request)}/jobs/{job_id}/model"
    )


@app.get("/jobs/{job_id}/exports", tags=["Download"])
async def list_exports(job_id: str):
    """Список всех экспортированных моделей"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    exports = []
    output_dir = job.output_dir
    
    if output_dir and output_dir.exists():
        for ply_file in sorted(output_dir.glob("*.ply")):
            exports.append({
                "filename": ply_file.name,
                "size_bytes": ply_file.stat().st_size,
                "created_at": datetime.fromtimestamp(ply_file.stat().st_mtime).isoformat()
            })
    
    return {"exports": exports}


@app.get("/jobs/{job_id}/exports/{filename}", tags=["Download"])
async def download_export(job_id: str, filename: str):
    """Скачать конкретный экспорт"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    file_path = job.output_dir / filename
    if not file_path.exists() or not file_path.suffix == '.ply':
        raise HTTPException(status_code=404, detail="Export not found")
    
    return FileResponse(
        str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


@app.post("/jobs/{job_id}/preview", response_model=PreviewInfo, tags=["Preview"])
async def create_preview(job_id: str, request: PreviewRequest = Body(default=PreviewRequest())):
    """Render a video preview of the trained model (flythrough along the
    scene's actual capture poses).

    This is a separate, explicit, optional request - never triggered
    automatically by training. Correctly handles the fact that a
    `colmap_images`/`hybrid_colmap_pointcloud` model lives in COLMAP's own
    (independently solved, differently oriented) coordinate frame, unlike
    `pointcloud_only`/`hybrid_known_poses` models - see preview_renderer.py.
    """
    job_manager = get_job_manager()

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed - nothing to preview yet")

    started = await job_manager.start_preview_render(job_id)
    if not started:
        raise HTTPException(status_code=409, detail="A preview render is already in progress for this job")

    asyncio.create_task(_run_preview_render(job_id, request))

    return PreviewInfo(job_id=job_id, status=PreviewStatus.RENDERING)


async def _run_preview_render(job_id: str, request: PreviewRequest):
    """Background task kicked off by create_preview(). Runs the actual (CPU/GPU-bound,
    blocking) rendering in a thread executor so it doesn't stall the event loop."""
    job_manager = get_job_manager()
    async with _preview_render_lock:
        job = await job_manager.get_job(job_id)
        if not job:
            return
        try:
            from . import preview_renderer

            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                lambda: preview_renderer.render_preview_video(
                    job,
                    job.preview_path,
                    num_waypoints=request.num_waypoints,
                    sub_steps=request.sub_steps,
                    path_smoothing=request.path_smoothing,
                    fps=request.fps,
                    resolution=request.resolution,
                ),
            )
            await job_manager.mark_preview_completed(job_id, stats)
        except Exception as e:
            await job_manager.mark_preview_failed(job_id, str(e))


@app.get("/jobs/{job_id}/preview", response_model=PreviewInfo, tags=["Preview"])
async def get_preview(job_id: str, http_request: Request):
    """Check status of a requested preview render, and get its download URL once ready."""
    job_manager = get_job_manager()

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    download_url = None
    if job.preview_status == PreviewStatus.COMPLETED and job.preview_path and job.preview_path.exists():
        download_url = f"{_public_base_url(http_request)}/jobs/{job_id}/preview/download"

    return PreviewInfo(
        job_id=job_id,
        status=job.preview_status,
        error=job.preview_error,
        download_url=download_url,
        stats=job.preview_stats,
    )


@app.get("/jobs/{job_id}/preview/download", tags=["Preview"])
async def download_preview(job_id: str):
    """Download the rendered preview video."""
    job_manager = get_job_manager()

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.preview_status != PreviewStatus.COMPLETED or not job.preview_path or not job.preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not available")

    return FileResponse(
        str(job.preview_path),
        filename=f"{job.name}_preview.mp4",
        media_type="video/mp4"
    )


@app.get("/jobs/{job_id}/camera_positions", tags=["Preview"])
async def get_camera_positions(job_id: str, count: int = Query(default=12, ge=1, le=100)):
    """A small, well-spread set of the job's actual capture camera poses -
    lets a human quickly pick a good starting viewpoint in an external 3DGS
    viewer loading the same exported .ply, instead of scrolling through
    hundreds of near-duplicate training frames. Cheap and synchronous - just
    reads camera pose files, no model loading or GPU rendering involved.
    Handles the same COLMAP-vs-original-frame distinction as the preview
    video (see preview_renderer.py's module docstring) - positions/vectors
    are returned in world space, the same frame as the exported .ply.
    """
    job_manager = get_job_manager()

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training not completed - no camera poses to offer yet")

    from . import preview_renderer

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: preview_renderer.get_camera_positions(job, count))
    except preview_renderer.PreviewError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"job_id": job_id, **result}


# ============== WebSocket API ==============

@app.websocket("/jobs/{job_id}/ws")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket для real-time обновлений прогресса"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return
    
    await websocket.accept()
    
    # Callback для отправки сообщений
    async def send_message(message: WebSocketMessage):
        try:
            await websocket.send_json(message.model_dump(mode='json'))
        except Exception:
            pass
    
    # Подписаться на обновления
    job_manager.subscribe(job_id, send_message)
    
    try:
        # Отправить текущее состояние
        await websocket.send_json({
            "type": "initial",
            "job_id": job_id,
            "data": job.to_info().model_dump(mode='json'),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Держать соединение открытым
        while True:
            try:
                # Ожидать сообщения от клиента (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Отправить ping
                await websocket.send_text("ping")
                
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(job_id, send_message)


# ============== Streaming logs ==============

@app.get("/jobs/{job_id}/logs", tags=["Logs"])
async def get_logs(job_id: str, tail: int = Query(default=100, le=1000)):
    """Получить последние логи задачи"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    log_file = job.job_dir / "log.txt"
    if not log_file.exists():
        return {"logs": []}
    
    async with aiofiles.open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = await f.readlines()
    
    return {"logs": lines[-tail:]}


# ============== Main entry point ==============

def main():
    """Запуск сервера"""
    import uvicorn
    
    config = get_config()
    
    uvicorn.run(
        "gs_server.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        workers=1  # Один воркер для shared state
    )


if __name__ == "__main__":
    main()
