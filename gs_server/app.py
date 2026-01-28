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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from .config import get_config, AppConfig
from .models import (
    JobStatus, JobInfo, JobProgress, CreateJobRequest, CreateJobResponse,
    JobListResponse, UploadProgressResponse, StartTrainingRequest,
    ModelInfo, ServerStatus, WebSocketMessage
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

# CORS для доступа из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def create_job(request: CreateJobRequest):
    """Создать новую задачу"""
    job_manager = get_job_manager()
    config = get_config()
    
    job = await job_manager.create_job(
        name=request.name,
        colmap_config=request.colmap_config,
        brush_config=request.brush_config,
        auto_start=request.auto_start
    )
    
    base_url = f"http://{config.server.host}:{config.server.port}"
    
    return CreateJobResponse(
        job_id=job.job_id,
        upload_url=f"{base_url}/jobs/{job.job_id}/upload",
        websocket_url=f"ws://{config.server.host}:{config.server.port}/jobs/{job.job_id}/ws"
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
async def start_job(job_id: str, request: Optional[StartTrainingRequest] = None):
    """Запустить обработку задачи вручную"""
    job_manager = get_job_manager()
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Обновить конфиг если передан
    if request and request.brush_config:
        job.config["brush"] = {**job.config.get("brush", {}), **request.brush_config}
    
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
async def get_model_info(job_id: str):
    """Получить информацию о модели"""
    job_manager = get_job_manager()
    config = get_config()
    
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
        download_url=f"http://{config.server.host}:{config.server.port}/jobs/{job_id}/model"
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
