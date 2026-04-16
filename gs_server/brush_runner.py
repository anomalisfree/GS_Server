"""
Brush Runner - Gaussian Splatting training via Brush
"""

import asyncio
import re
import os
import time
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .models import JobStatus, TrainingProgress
from .config import get_config

if TYPE_CHECKING:
    from .job_manager import JobManager, Job

logger = logging.getLogger(__name__)


class BrushRunner:
    """Runs Brush training"""
    
    def __init__(self, job_manager: "JobManager"):
        self._job_manager = job_manager
        self._config = get_config()
        self._brush_dir = self._config.paths.brush_dir
        
    async def run(self, job: "Job"):
        """Start training"""
        job_id = job.job_id
        colmap_dir = job.colmap_dir
        output_dir = job.output_dir
        
        # Определить путь к данным COLMAP
        # Brush принимает либо dense/sparse, либо undistorted images
        dataset_path = colmap_dir / "dense"
        if not dataset_path.exists():
            dataset_path = colmap_dir / "sparse" / "0"
        if not dataset_path.exists():
            dataset_path = colmap_dir / "sparse"
            
        if not dataset_path.exists():
            await self._job_manager.mark_failed(job_id, "COLMAP data not found")
            return
        
        # Настройки тренировки
        brush_config = self._config.brush
        job_brush_config = job.config.get("brush", {})
        
        total_steps = job_brush_config.get("total_steps", brush_config.total_steps)
        max_resolution = job_brush_config.get("max_resolution", brush_config.max_resolution)
        eval_every = job_brush_config.get("eval_every", brush_config.eval_every)
        export_every = job_brush_config.get("export_every", brush_config.export_every)
        refine_every = job_brush_config.get("refine_every", brush_config.refine_every)
        
        # Путь для экспорта модели
        export_path = output_dir / "model_{iter}.ply"
        final_model_path = output_dir / f"model_{total_steps}.ply"
        
        await self._job_manager.update_job_progress(
            job_id,
            status=JobStatus.TRAINING,
            message="Starting Brush training...",
            overall_progress=50.0,
            training_progress=TrainingProgress(
                current_step=0,
                total_steps=total_steps,
                message="Initialization"
            )
        )
        
        try:
            # Собрать команду - используем собранный exe напрямую
            brush_exe = self._brush_dir / "target" / "release" / "brush.exe"
            if not brush_exe.exists():
                # Fallback на cargo run если exe не найден
                brush_exe = None
            
            if brush_exe:
                cmd = [
                    str(brush_exe),
                    str(dataset_path),
                ]
            else:
                cmd = [
                    "cargo", "run", "--release", "--",
                    str(dataset_path),
                ]
            
            cmd.extend([
                "--total-train-iters", str(total_steps),
                "--max-resolution", str(max_resolution),
                "--eval-every", str(eval_every),
                "--export-every", str(export_every),
                "--refine-every", str(refine_every),
                "--export-path", str(output_dir),
                "--export-name", "model_{iter}.ply",
            ])
            
            # Добавить дополнительные параметры из конфига
            if lr_mean := job_brush_config.get("lr_mean"):
                cmd.extend(["--lr-mean", str(lr_mean)])
            if lr_opac := job_brush_config.get("lr_opac"):
                cmd.extend(["--lr-opac", str(lr_opac)])
            if ssim_weight := job_brush_config.get("ssim_weight"):
                cmd.extend(["--ssim-weight", str(ssim_weight)])
            if max_splats := job_brush_config.get("max_splats"):
                cmd.extend(["--max-splats", str(max_splats)])
            
            # Устанавливаем переменные окружения для отключения интерактивного вывода
            # и включения логов
            env = os.environ.copy()
            env.update({
                "TERM": "dumb",  # Отключает цветной/интерактивный вывод
                "NO_COLOR": "1",  # Некоторые программы используют это
                "CI": "1",  # Некоторые программы отключают интерактивный режим в CI
                "RUST_LOG": "brush_cli=info,brush_train=info,info",  # Включаем info логи для brush модулей
                "RUST_BACKTRACE": "1",  # Для отладки ошибок
            })
            
            cmd_str = ' '.join(str(c) for c in cmd)
            logger.info(f"Running Brush: {cmd_str}")
            print(f"[BRUSH] Running: {cmd_str}")
                
            # Запустить процесс
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._brush_dir),
                env=env
            )
            
            start_time = time.time()
            current_step = 0
            splat_count = 0
            last_psnr = None
            last_ssim = None
            last_export_step = 0
            last_export_path = None
            last_progress_update = start_time
            
            # Флаг для остановки мониторинга
            monitoring = True
            
            # Запускаем фоновую задачу для мониторинга экспортированных файлов
            async def monitor_exports():
                nonlocal last_export_step, last_export_path, current_step, last_progress_update
                while monitoring:
                    await asyncio.sleep(2)  # Проверяем каждые 2 секунды
                    
                    # Ищем экспортированные файлы
                    ply_files = sorted(output_dir.glob("model_*.ply"), reverse=True)
                    for ply_file in ply_files:
                        match = re.search(r'model_(\d+)\.ply', ply_file.name)
                        if match:
                            step = int(match.group(1))
                            if step > last_export_step:
                                last_export_step = step
                                last_export_path = str(ply_file)
                                # Если экспорт есть, значит тренировка дошла минимум до этого шага
                                if step > current_step:
                                    current_step = step
                                    logger.info(f"Detected export at step {step}: {ply_file}")
                                    print(f"[BRUSH] Detected export at step {step}")
                                    
                                    # Обновляем прогресс
                                    elapsed = time.time() - start_time
                                    steps_per_sec = current_step / elapsed if elapsed > 0 else 0
                                    eta = (total_steps - current_step) / steps_per_sec if steps_per_sec > 0 else None
                                    train_progress = (current_step / total_steps) if total_steps > 0 else 0
                                    overall = 50.0 + train_progress * 50.0
                                    
                                    await self._job_manager.update_job_progress(
                                        job_id,
                                        training_progress=TrainingProgress(
                                            current_step=current_step,
                                            total_steps=total_steps,
                                            steps_per_second=round(steps_per_sec, 2),
                                            elapsed_seconds=round(elapsed, 1),
                                            eta_seconds=round(eta, 1) if eta else None,
                                            psnr=last_psnr,
                                            ssim=last_ssim,
                                            splat_count=splat_count,
                                            last_export_step=last_export_step,
                                            last_export_path=last_export_path
                                        ),
                                        overall_progress=overall,
                                        message=f"Step {current_step}/{total_steps}"
                                    )
                                    last_progress_update = time.time()
                            break  # Берём только последний (самый большой номер)
            
            monitor_task = asyncio.create_task(monitor_exports())
            
            try:
                async for line in process.stdout:
                    text = line.decode('utf-8', errors='ignore').strip()
                    # Убираем ANSI escape последовательности
                    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
                    text = re.sub(r'\r', '', text)
                    
                    if not text:
                        continue
                    
                    # Логируем весь вывод для отладки
                    print(f"[BRUSH] {text}")
                    
                    # Проверить отмену
                    current_job = await self._job_manager.get_job(job_id)
                    if current_job and current_job.status == JobStatus.CANCELLED:
                        process.terminate()
                        monitoring = False
                        monitor_task.cancel()
                        return
                    
                    # Парсинг вывода Brush
                    # Brush использует log::info! для важных сообщений:
                    # - "Refine iter {iter}, {cur_splat_count} splats."
                    # - "Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}"
                    
                    # Refine step с количеством сплатов
                    if match := re.search(r'Refine iter (\d+),\s*(\d+)\s*splats', text, re.IGNORECASE):
                        current_step = int(match.group(1))
                        splat_count = int(match.group(2))
                        logger.info(f"Parsed refine: step={current_step}, splats={splat_count}")
                    
                    # Eval результат с PSNR и SSIM
                    elif match := re.search(r'Eval iter (\d+):\s*PSNR\s*([\d.]+),?\s*ssim\s*([\d.]+)', text, re.IGNORECASE):
                        current_step = int(match.group(1))
                        last_psnr = float(match.group(2))
                        last_ssim = float(match.group(3))
                        logger.info(f"Parsed eval: step={current_step}, PSNR={last_psnr}, SSIM={last_ssim}")
                    
                    # indicatif progress bar pattern: число/число
                    elif match := re.search(r'\b(\d+)/(\d+)\b.*Steps', text):
                        step = int(match.group(1))
                        total = int(match.group(2))
                        # Проверяем что это похоже на шаг тренировки
                        if total == total_steps and step > current_step:
                            current_step = step
                    
                    # PSNR отдельно (на случай другого формата)
                    if "PSNR" in text.upper():
                        if match := re.search(r'PSNR[:\s]*([\d.]+)', text, re.IGNORECASE):
                            last_psnr = float(match.group(1))
                        
                    # SSIM отдельно
                    if "ssim" in text.lower():
                        if match := re.search(r'ssim[:\s]*([\d.]+)', text, re.IGNORECASE):
                            last_ssim = float(match.group(1))
                        
                    # Splat count
                    if "splat" in text.lower():
                        if match := re.search(r'(\d+)\s*splats?', text, re.IGNORECASE):
                            splat_count = int(match.group(1))
                    
                    # Обновляем прогресс периодически (не чаще раза в секунду)
                    now = time.time()
                    if now - last_progress_update >= 1.0:
                        elapsed = now - start_time
                        steps_per_sec = current_step / elapsed if elapsed > 0 and current_step > 0 else 0
                        eta = (total_steps - current_step) / steps_per_sec if steps_per_sec > 0 else None
                        train_progress = (current_step / total_steps) if total_steps > 0 else 0
                        overall = 50.0 + train_progress * 50.0
                        
                        await self._job_manager.update_job_progress(
                            job_id,
                            training_progress=TrainingProgress(
                                current_step=current_step,
                                total_steps=total_steps,
                                steps_per_second=round(steps_per_sec, 2),
                                elapsed_seconds=round(elapsed, 1),
                                eta_seconds=round(eta, 1) if eta else None,
                                psnr=last_psnr,
                                ssim=last_ssim,
                                splat_count=splat_count,
                                last_export_step=last_export_step,
                                last_export_path=last_export_path
                            ),
                            overall_progress=overall,
                            message=f"Step {current_step}/{total_steps}"
                        )
                        last_progress_update = now
                
            finally:
                monitoring = False
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            await process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"Brush training failed with code {process.returncode}")
            
            # Найти финальную модель
            final_model = None
            final_size = 0
            
            # Ищем последний экспортированный файл
            for ply_file in sorted(output_dir.glob("model_*.ply"), reverse=True):
                final_model = str(ply_file)
                final_size = ply_file.stat().st_size
                break
            
            if not final_model:
                raise RuntimeError("Model was not exported")
            
            # Обновить финальный прогресс тренировки
            await self._job_manager.update_job_progress(
                job_id,
                training_progress=TrainingProgress(
                    current_step=total_steps,
                    total_steps=total_steps,
                    steps_per_second=0,
                    elapsed_seconds=round(time.time() - start_time, 1),
                    eta_seconds=0,
                    psnr=last_psnr,
                    ssim=last_ssim,
                    splat_count=splat_count,
                    last_export_step=total_steps,
                    last_export_path=final_model
                )
            )
            
            # Отметить как завершенную
            await self._job_manager.mark_completed(job_id, final_model, final_size)
            
        except Exception as e:
            logger.error(f"Brush training error: {e}")
            await self._job_manager.update_job_progress(
                job_id,
                status=JobStatus.TRAINING_FAILED,
                error=str(e),
                message=f"Training error: {e}"
            )
