"""
COLMAP Runner - automatic reconstruction pipeline via COLMAP
"""

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .models import JobStatus, ColmapProgress, ColmapStage
from .config import get_config

if TYPE_CHECKING:
    from .job_manager import JobManager, Job


class ColmapRunner:
    """Runs COLMAP for image processing"""
    
    def __init__(self, job_manager: "JobManager"):
        self._job_manager = job_manager
        self._config = get_config()
        self._colmap_exe = str(self._config.paths.colmap_exe)
        
    async def run(self, job: "Job"):
        """Run full COLMAP pipeline"""
        job_id = job.job_id
        images_dir = job.images_dir
        colmap_dir = job.colmap_dir
        
        # Создать директории для COLMAP
        database_path = colmap_dir / "database.db"
        sparse_dir = colmap_dir / "sparse"
        dense_dir = colmap_dir / "dense"
        
        sparse_dir.mkdir(exist_ok=True)
        dense_dir.mkdir(exist_ok=True)
        
        colmap_config = self._config.colmap
        job_colmap_config = job.config.get("colmap", {})
        
        # Объединить конфиги
        use_gpu = job_colmap_config.get("use_gpu", colmap_config.use_gpu)
        camera_model = job_colmap_config.get("camera_model", colmap_config.camera_model)
        single_camera = job_colmap_config.get("single_camera", colmap_config.single_camera)
        max_image_size = job_colmap_config.get("max_image_size", colmap_config.max_image_size)
        
        await self._job_manager.update_job_progress(
            job_id,
            status=JobStatus.COLMAP_RUNNING,
            message="Starting COLMAP...",
            overall_progress=5.0,
            colmap_progress=ColmapProgress(
                stage=ColmapStage.FEATURE_EXTRACTION,
                message="Initialization"
            )
        )
        
        try:
            # 1. Feature Extraction
            await self._run_feature_extraction(
                job_id, images_dir, database_path,
                use_gpu, camera_model, single_camera, max_image_size
            )
            
            if (await self._job_manager.get_job(job_id)).status == JobStatus.CANCELLED:
                return
            
            # 2. Feature Matching
            await self._run_feature_matching(
                job_id, database_path, use_gpu
            )
            
            if (await self._job_manager.get_job(job_id)).status == JobStatus.CANCELLED:
                return
            
            # 3. Sparse Reconstruction (Mapper)
            await self._run_mapper(
                job_id, database_path, images_dir, sparse_dir
            )
            
            if (await self._job_manager.get_job(job_id)).status == JobStatus.CANCELLED:
                return
            
            # 4. Image Undistortion
            await self._run_undistortion(
                job_id, images_dir, sparse_dir, dense_dir
            )
            
            # Done!
            await self._job_manager.update_job_progress(
                job_id,
                status=JobStatus.COLMAP_DONE,
                message="COLMAP completed successfully",
                overall_progress=50.0,
                colmap_progress=ColmapProgress(
                    stage=ColmapStage.IMAGE_UNDISTORTION,
                    stage_progress=100.0,
                    message="Completed"
                )
            )
            
        except Exception as e:
            await self._job_manager.update_job_progress(
                job_id,
                status=JobStatus.COLMAP_FAILED,
                error=str(e),
                message=f"COLMAP error: {e}"
            )
    
    async def _run_feature_extraction(
        self, 
        job_id: str,
        images_dir: Path,
        database_path: Path,
        use_gpu: bool,
        camera_model: str,
        single_camera: bool,
        max_image_size: int
    ):
        """Stage 1: Feature extraction"""
        await self._job_manager.update_job_progress(
            job_id,
            colmap_progress=ColmapProgress(
                stage=ColmapStage.FEATURE_EXTRACTION,
                stage_progress=0.0,
                message="Extracting features from images..."
            ),
            overall_progress=6.0
        )
        
        cmd = [
            self._colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--SiftExtraction.max_image_size", str(max_image_size),
            "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
            "--SiftExtraction.max_num_features", str(self._config.colmap.sift_max_features),
        ]
        
        await self._run_command(
            job_id, cmd, ColmapStage.FEATURE_EXTRACTION,
            base_progress=6.0, progress_range=12.0
        )
    
    async def _run_feature_matching(
        self,
        job_id: str,
        database_path: Path,
        use_gpu: bool
    ):
        """Stage 2: Feature matching"""
        await self._job_manager.update_job_progress(
            job_id,
            colmap_progress=ColmapProgress(
                stage=ColmapStage.FEATURE_MATCHING,
                stage_progress=0.0,
                message="Matching features..."
            ),
            overall_progress=18.0
        )
        
        matcher = self._config.colmap.matcher_type
        
        cmd = [
            self._colmap_exe, f"{matcher}_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
        ]
        
        await self._run_command(
            job_id, cmd, ColmapStage.FEATURE_MATCHING,
            base_progress=18.0, progress_range=12.0
        )
    
    async def _run_mapper(
        self,
        job_id: str,
        database_path: Path,
        images_dir: Path,
        sparse_dir: Path
    ):
        """Stage 3: Sparse reconstruction"""
        await self._job_manager.update_job_progress(
            job_id,
            colmap_progress=ColmapProgress(
                stage=ColmapStage.SPARSE_RECONSTRUCTION,
                stage_progress=0.0,
                message="Building sparse model..."
            ),
            overall_progress=30.0
        )
        
        cmd = [
            self._colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ]
        
        await self._run_command(
            job_id, cmd, ColmapStage.SPARSE_RECONSTRUCTION,
            base_progress=30.0, progress_range=15.0
        )
    
    async def _run_undistortion(
        self,
        job_id: str,
        images_dir: Path,
        sparse_dir: Path,
        dense_dir: Path
    ):
        """Stage 4: Image undistortion"""
        await self._job_manager.update_job_progress(
            job_id,
            colmap_progress=ColmapProgress(
                stage=ColmapStage.IMAGE_UNDISTORTION,
                stage_progress=0.0,
                message="Undistorting images..."
            ),
            overall_progress=45.0
        )
        
        # Найти модель (обычно в папке 0)
        model_path = sparse_dir / "0"
        if not model_path.exists():
            # Попробовать найти любую папку с моделью
            for p in sparse_dir.iterdir():
                if p.is_dir() and (p / "cameras.bin").exists():
                    model_path = p
                    break
        
        cmd = [
            self._colmap_exe, "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(model_path),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ]
        
        await self._run_command(
            job_id, cmd, ColmapStage.IMAGE_UNDISTORTION,
            base_progress=45.0, progress_range=5.0
        )
    
    async def _run_command(
        self,
        job_id: str,
        cmd: list,
        stage: ColmapStage,
        base_progress: float,
        progress_range: float
    ):
        """Run COLMAP command with output parsing"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Логировать команду
        cmd_str = ' '.join(str(c) for c in cmd)
        logger.info(f"Running COLMAP: {cmd_str}")
        print(f"[COLMAP] Running: {cmd_str}")
        
        # Собираем весь вывод для диагностики
        all_output = []
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        images_processed = 0
        total_images = 0
        features_extracted = 0
        matches_found = 0
        registered = 0
        points_3d = 0
        
        async for line in process.stdout:
            text = line.decode('utf-8', errors='ignore').strip()
            if not text:
                continue
            
            # Собираем весь вывод
            all_output.append(text)
            print(f"[COLMAP] {text}")
                
            # Парсинг прогресса
            stage_progress = 0.0
            
            # Feature extraction progress
            if match := re.search(r'Processing image \[(\d+)/(\d+)\]', text):
                images_processed = int(match.group(1))
                total_images = int(match.group(2))
                stage_progress = (images_processed / total_images) * 100 if total_images > 0 else 0
                
            elif match := re.search(r'Extracted (\d+) features', text):
                features_extracted += int(match.group(1))
                
            # Matching progress  
            elif match := re.search(r'Matching block \[(\d+)/(\d+)', text):
                cur = int(match.group(1))
                total = int(match.group(2))
                stage_progress = (cur / total) * 100 if total > 0 else 0
                
            elif match := re.search(r'Found (\d+) matches', text):
                matches_found += int(match.group(1))
                
            # Mapper progress
            elif match := re.search(r'Registering image #\d+ \((\d+)\)', text):
                registered = int(match.group(1))
                
            elif match := re.search(r'Points3D: (\d+)', text):
                points_3d = int(match.group(1))
                
            # Undistortion progress
            elif match := re.search(r'Undistorting image \[(\d+)/(\d+)\]', text):
                cur = int(match.group(1))
                total = int(match.group(2))
                stage_progress = (cur / total) * 100 if total > 0 else 0
            
            # Обновить прогресс
            if stage_progress > 0 or images_processed > 0:
                overall = base_progress + (stage_progress / 100) * progress_range
                
                await self._job_manager.update_job_progress(
                    job_id,
                    colmap_progress=ColmapProgress(
                        stage=stage,
                        stage_progress=stage_progress,
                        images_processed=images_processed,
                        total_images=total_images,
                        features_extracted=features_extracted,
                        matches_found=matches_found,
                        registered_images=registered,
                        points_3d=points_3d,
                        message=text[:200]
                    ),
                    overall_progress=overall
                )
        
        await process.wait()
        
        if process.returncode != 0:
            # Получить последние строки вывода для диагностики
            last_output = '\n'.join(all_output[-20:]) if all_output else "No output"
            error_msg = f"COLMAP {stage.value} failed with code {process.returncode}\nOutput:\n{last_output}"
            logger.error(error_msg)
            print(f"[COLMAP ERROR] {error_msg}")
            raise RuntimeError(error_msg)
