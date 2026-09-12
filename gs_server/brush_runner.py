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
from .floater_filter import filter_needle_floaters, filter_isolated_clusters

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

        # Для режима без COLMAP можно передать готовый датасет напрямую
        dataset_override = job.config.get("dataset_path")
        if dataset_override:
            dataset_path = Path(dataset_override)
        else:
            # Определить путь к данным COLMAP
            # Brush принимает либо dense/sparse, либо undistorted images
            dataset_path = colmap_dir / "dense"
            if not dataset_path.exists():
                dataset_path = colmap_dir / "sparse" / "0"
            if not dataset_path.exists():
                dataset_path = colmap_dir / "sparse"
            
        if not dataset_path.exists():
            await self._job_manager.mark_failed(job_id, f"Dataset path not found: {dataset_path}")
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
            # Сначала проверяем явно заданный путь из конфига (GS_BRUSH_EXE)
            explicit_exe = self._config.paths.brush_exe
            if explicit_exe and explicit_exe.exists():
                resolved_exe = explicit_exe
            else:
                # Fallback: ищем binary в brush_dir/target/release/
                # На Windows – brush.exe, на Linux – brush
                for candidate in ("brush.exe", "brush"):
                    p = self._brush_dir / "target" / "release" / candidate
                    if p.exists():
                        resolved_exe = p
                        break
                else:
                    resolved_exe = None

            if resolved_exe:
                cmd = [
                    str(resolved_exe),
                    str(dataset_path),
                ]
            else:
                cmd = [
                    "cargo", "run", "--release", "--",
                    str(dataset_path),
                ]
            
            cmd.extend([
                "--total-steps", str(total_steps),
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
            if max_splats := job_brush_config.get("max_splats"):
                cmd.extend(["--max-splats", str(max_splats)])

            # Growth / noise control
            growth_grad = job_brush_config.get("growth_grad_threshold", brush_config.growth_grad_threshold)
            growth_frac = job_brush_config.get("growth_select_fraction", brush_config.growth_select_fraction)
            growth_stop = job_brush_config.get("stop_growth_at", brush_config.stop_growth_at)
            opac_decay   = job_brush_config.get("opac_decay",   brush_config.opac_decay)
            scale_decay  = job_brush_config.get("scale_decay",  brush_config.scale_decay)
            ssim_weight  = job_brush_config.get("ssim_weight",  brush_config.ssim_weight)
            cmd.extend([
                "--growth-grad-threshold",   str(growth_grad),
                "--growth-select-fraction",  str(growth_frac),
                "--growth-stop-iter",        str(growth_stop),
                "--ssim-weight",             str(ssim_weight),
                # Brush's real flag names/units for what this config calls opac_decay/
                # scale_decay — previously never passed at all, see config.py note.
                "--opac-loss-weight",        str(opac_decay),
                "--scale-loss-weight",       str(scale_decay),
            ])
            
            # Устанавливаем переменные окружения для отключения интерактивного вывода
            # и включения логов
            env = os.environ.copy()
            env.update({
                "TERM": "dumb",  # Отключает цветной/интерактивный вывод
                "NO_COLOR": "1",  # Некоторые программы используют это
                "CI": "1",  # Некоторые программы отключают интерактивный режим в CI
                # Не включаем глобальный info, иначе некоторые модули печатают
                # очень длинные строки (device features), которые ломают readline.
                "RUST_LOG": "warn,brush_cli=info,brush_train=info,brush_process=info,brush_dataset=info",
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
                cwd=str(output_dir),  # Run from output dir — brush_dir may not exist on server
                env=env,
                limit=8 * 1024 * 1024,
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
            
            # Check if training completed successfully by looking for exported models.
            # Brush sometimes crashes (SIGSEGV, code -11) during WGPU/Vulkan shutdown
            # AFTER the final model is already exported — this is still a success.
            model_files = sorted(output_dir.glob("model_*.ply"), reverse=True)
            last_exported_step = 0
            if model_files:
                m = re.search(r'model_0*(\d+)\.ply', model_files[0].name)
                if m:
                    last_exported_step = int(m.group(1))

            if process.returncode != 0:
                is_signal_crash = process.returncode < 0  # negative = killed by OS signal
                model_complete = last_exported_step >= int(total_steps * 0.9)  # ≥90% done
                if is_signal_crash and model_complete:
                    logger.warning(
                        f"Brush exited with signal {-process.returncode} after exporting "
                        f"model at step {last_exported_step}/{total_steps} — treating as success"
                    )
                    print(f"[BRUSH] Signal crash after export at step {last_exported_step} — OK")
                else:
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

            # Убрать needle-выбросы на краях объекта (см. floater_filter.py) —
            # хирургически, в отличие от opac/scale decay, который не снижает
            # анизотропию выживших гауссианов, а просто прорежает их количество.
            filtered_path = Path(final_model).with_name(Path(final_model).stem + "_filtered.ply")

            floater_enabled = job_brush_config.get("floater_filter_enabled", brush_config.floater_filter_enabled)
            if floater_enabled:
                aniso_threshold = job_brush_config.get(
                    "floater_filter_aniso_threshold", brush_config.floater_filter_aniso_threshold
                )
                size_multiplier = job_brush_config.get(
                    "floater_filter_size_multiplier", brush_config.floater_filter_size_multiplier
                )
                try:
                    stats = filter_needle_floaters(final_model, str(filtered_path), aniso_threshold, size_multiplier)
                    if not stats.get("skipped"):
                        print(
                            f"[BRUSH] Floater filter removed {stats['removed_count']} "
                            f"({stats['removed_pct']}%) needle splats "
                            f"(median splat size {stats['median_maxdim']:.4f}, "
                            f"threshold {stats['effective_min_size']:.4f})"
                        )
                        final_model = str(filtered_path)
                        final_size = filtered_path.stat().st_size
                except Exception as e:
                    logger.warning(f"Floater filter failed, keeping unfiltered model: {e}")

            # Remove scattered debris clusters spatially disconnected from the
            # main scanned structure (see floater_filter.py's docstring for why
            # this is a separate pass from the needle filter above - shape-based
            # filtering has no notion of spatial isolation, so it misses this
            # population entirely). Chains onto whatever final_model currently
            # is (needle-filtered or not) and always writes the canonical
            # "_filtered.ply" output.
            cluster_enabled = job_brush_config.get("cluster_filter_enabled", brush_config.cluster_filter_enabled)
            if cluster_enabled:
                radius_multiplier = job_brush_config.get(
                    "cluster_filter_radius_multiplier", brush_config.cluster_filter_radius_multiplier
                )
                try:
                    stats = filter_isolated_clusters(final_model, str(filtered_path), radius_multiplier=radius_multiplier)
                    if not stats.get("skipped"):
                        print(
                            f"[BRUSH] Cluster filter removed {stats['removed_count']} "
                            f"({stats['removed_pct']}%) isolated splats "
                            f"({stats['n_components']} components, main component "
                            f"{100 * stats['main_component_frac']:.1f}% of model)"
                        )
                        final_model = str(filtered_path)
                        final_size = filtered_path.stat().st_size
                except Exception as e:
                    logger.warning(f"Cluster filter failed, keeping model as-is: {e}")

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
