"""
Прямой запуск COLMAP + Brush на высококачественных фото из data/images.

Pipeline:
  1. COLMAP feature extraction
  2. COLMAP exhaustive matching
  3. COLMAP mapper (sparse reconstruction)
  4. COLMAP image_undistorter (dense/undistorted images)
  5. Brush training (GS) на результате COLMAP

Выход: data/NicoOffice_hires_exports/
"""

import subprocess
import sys
import time
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────────────────────
BASE        = Path(r"D:\VSProjects\GS_Sever")
COLMAP_EXE  = BASE / "colmap" / "bin" / "colmap.exe"
BRUSH_EXE   = BASE / "brush" / "target" / "release" / "brush.exe"
IMAGES_DIR  = BASE / "data" / "images"
WORK_DIR    = BASE / "data" / "NicoOffice_hires"   # рабочая директория COLMAP
EXPORTS_DIR = BASE / "data" / "NicoOffice_hires_exports"

DATABASE    = WORK_DIR / "database.db"
SPARSE_DIR  = WORK_DIR / "sparse"
DENSE_DIR   = WORK_DIR / "dense"

# ── Настройки COLMAP ───────────────────────────────────────────────────────────
CAMERA_MODEL  = "OPENCV"
SINGLE_CAMERA = True       # все кадры с одной камеры (Leica scanner)
MAX_IMAGE_SIZE = 3200
MAX_FEATURES   = 8192
USE_GPU        = True      # поставить False если нет CUDA

# Sequential matching: перекрытие ±N соседних кадров
SEQ_OVERLAP = 20           # каждый кадр матчится с ближайшими 20-ю

# Mapper: расслабленные пороги для blur/low-texture
MAPPER_INIT_MIN_INLIERS   = 15    # default 100 — снижаем для blur
MAPPER_MIN_MATCHES        = 5     # default 15
MAPPER_ABS_POSE_MIN_INLIERS = 15  # default 30
MAPPER_ABS_POSE_MIN_RATIO = 0.1   # default 0.25

# ── Настройки Brush ────────────────────────────────────────────────────────────
TOTAL_STEPS   = 30000
MAX_RESOLUTION = 1920
EVAL_EVERY    = 1000
EXPORT_EVERY  = 2500
REFINE_EVERY  = 200

# ──────────────────────────────────────────────────────────────────────────────

def run(cmd: list, label: str):
    """Запустить команду с выводом в консоль, завершить при ошибке."""
    print(f"\n{'='*60}")
    print(f"[{label}] {' '.join(str(c) for c in cmd)}")
    print('='*60)
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[{label}] FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        sys.exit(result.returncode)
    print(f"\n[{label}] Done in {elapsed:.0f}s")


def main():
    # Проверить наличие инструментов
    for exe in [COLMAP_EXE, BRUSH_EXE]:
        if not exe.exists():
            print(f"[ERROR] Not found: {exe}")
            sys.exit(1)

    # Проверить папку с изображениями
    images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    if not images:
        print(f"[ERROR] No images in {IMAGES_DIR}")
        sys.exit(1)
    print(f"Found {len(images)} images in {IMAGES_DIR}")

    # Сбросить плохой предыдущий результат (sparse/0 с 2 кадрами)
    bad_sparse = SPARSE_DIR / "0"
    if bad_sparse.exists():
        import shutil
        # Проверить — если только 2 изображения, удалить и переделать
        cameras_bin = bad_sparse / "cameras.bin"
        images_bin  = bad_sparse / "images.bin"
        if cameras_bin.exists() and images_bin.stat().st_size < 5000:
            print("[INFO] Removing bad sparse/0 (only 2 images), will re-run mapper")
            shutil.rmtree(bad_sparse)
        if (WORK_DIR / "dense").exists():
            shutil.rmtree(WORK_DIR / "dense")
            (WORK_DIR / "dense").mkdir()

    # Создать рабочие папки
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    gpu_flag = "1" if USE_GPU else "0"

    # ── Шаг 1: Feature Extraction ─────────────────────────────────────────────
    if DATABASE.exists():
        print(f"[INFO] database.db already exists — re-using extracted features")
    else:
        run([
            str(COLMAP_EXE), "feature_extractor",
            "--database_path", str(DATABASE),
            "--image_path",    str(IMAGES_DIR),
            "--ImageReader.camera_model",  CAMERA_MODEL,
            "--ImageReader.single_camera", "1" if SINGLE_CAMERA else "0",
            "--SiftExtraction.max_image_size", str(MAX_IMAGE_SIZE),
            "--SiftExtraction.max_num_features", str(MAX_FEATURES),
            "--FeatureExtraction.use_gpu", gpu_flag,
        ], "COLMAP feature_extractor")

    # ── Шаг 2: Sequential Matching ────────────────────────────────────────────
    # Sequential: матчим каждый кадр с ±SEQ_OVERLAP соседями по индексу.
    # Подходит для панорамных sweep-последовательностей Leica scanner.
    # Флаг сбрасывается если пересчитываем (старые exhaustive-матчи несовместимы).
    match_done_flag = WORK_DIR / ".seq_matching_done"
    if match_done_flag.exists():
        print(f"[INFO] Sequential matching already done, skipping")
    else:
        run([
            str(COLMAP_EXE), "sequential_matcher",
            "--database_path", str(DATABASE),
            "--FeatureMatching.use_gpu", gpu_flag,
            "--SequentialMatching.overlap", str(SEQ_OVERLAP),
            "--SequentialMatching.loop_detection", "0",
        ], "COLMAP sequential_matcher")
        match_done_flag.touch()

    # ── Шаг 3: Mapper (Sparse Reconstruction) ────────────────────────────────
    model_dir = SPARSE_DIR / "0"
    if model_dir.exists() and (model_dir / "cameras.bin").exists():
        print(f"[INFO] Sparse model already exists, skipping mapper")
    else:
        run([
            str(COLMAP_EXE), "mapper",
            "--database_path", str(DATABASE),
            "--image_path",    str(IMAGES_DIR),
            "--output_path",   str(SPARSE_DIR),
            # Расслабленные пороги для blur / low-texture изображений
            "--Mapper.init_min_num_inliers",      str(MAPPER_INIT_MIN_INLIERS),
            "--Mapper.min_num_matches",           str(MAPPER_MIN_MATCHES),
            "--Mapper.abs_pose_min_num_inliers",  str(MAPPER_ABS_POSE_MIN_INLIERS),
            "--Mapper.abs_pose_min_inlier_ratio", str(MAPPER_ABS_POSE_MIN_RATIO),
        ], "COLMAP mapper")
        # Определить реальную папку модели
        model_dir = SPARSE_DIR / "0"
        if not model_dir.exists():
            for d in sorted(SPARSE_DIR.iterdir()):
                if d.is_dir() and (d / "cameras.bin").exists():
                    model_dir = d
                    break

    if not model_dir.exists():
        print("[ERROR] COLMAP mapper produced no output. Check images.")
        sys.exit(1)

    print(f"[INFO] Using sparse model: {model_dir}")

    # ── Шаг 4: Image Undistortion ──────────────────────────────────────────────
    undist_images = DENSE_DIR / "images"
    if undist_images.exists() and any(undist_images.iterdir()):
        print(f"[INFO] Undistorted images already exist, skipping undistortion")
    else:
        run([
            str(COLMAP_EXE), "image_undistorter",
            "--image_path",  str(IMAGES_DIR),
            "--input_path",  str(model_dir),
            "--output_path", str(DENSE_DIR),
            "--output_type", "COLMAP",
        ], "COLMAP image_undistorter")

    # DENSE_DIR содержит images/ sparse/ — это и есть вход для Brush
    dataset_path = DENSE_DIR
    if not (dataset_path / "sparse").exists():
        # fallback: sparse/0
        dataset_path = model_dir
        print(f"[WARN] dense not found, using sparse model directly: {dataset_path}")

    # ── Шаг 5: Brush Training ─────────────────────────────────────────────────
    run([
        str(BRUSH_EXE),
        str(dataset_path),
        "--total-train-iters", str(TOTAL_STEPS),
        "--max-resolution",    str(MAX_RESOLUTION),
        "--eval-every",        str(EVAL_EVERY),
        "--export-every",      str(EXPORT_EVERY),
        "--refine-every",      str(REFINE_EVERY),
        "--export-path",       str(EXPORTS_DIR),
        "--export-name",       "model_{iter}.ply",
    ], "Brush training")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Exports: {EXPORTS_DIR}")
    print('='*60)


if __name__ == "__main__":
    main()
