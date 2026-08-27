"""Запуск полного pipeline на данных GS_20260415_114533"""
import requests
import time
import os
import sys

BASE_URL = "http://localhost:8080"
IMAGES_DIR = r"D:\VSProjects\GS_Sever\GS_20260416_175706\images"

def main():
    # 1. Проверить сервер
    print("Connecting to server...")
    try:
        r = requests.get(f"{BASE_URL}/status")
        r.raise_for_status()
        print(f"Server OK: {r.json()}")
    except Exception as e:
        print(f"Server not available: {e}")
        sys.exit(1)

    # 2. Создать job
    print("\nCreating job...")
    r = requests.post(f"{BASE_URL}/jobs", json={
        "name": "GS_20260416_pipeline",
        "auto_start": True,
        "brush_config": {"total_steps": 30000}
    })
    r.raise_for_status()
    job = r.json()
    job_id = job["job_id"]
    print(f"Job created: {job_id}")

    # 3. Собрать изображения
    images = sorted([
        os.path.join(IMAGES_DIR, f) 
        for f in os.listdir(IMAGES_DIR) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"Found {len(images)} images")

    # 4. Загрузить пачками по 50
    batch_size = 50
    total_uploaded = 0
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        files = []
        opened = []
        for path in batch:
            f = open(path, "rb")
            opened.append(f)
            files.append(("files", (os.path.basename(path), f)))
        
        r = requests.post(f"{BASE_URL}/jobs/{job_id}/upload", files=files)
        r.raise_for_status()
        
        for f in opened:
            f.close()
        
        total_uploaded += len(batch)
        print(f"Uploaded {total_uploaded}/{len(images)} images")

    # 5. Завершить загрузку -> auto_start запустит pipeline
    print("\nCompleting upload (starts pipeline)...")
    r = requests.post(f"{BASE_URL}/jobs/{job_id}/upload/complete")
    r.raise_for_status()
    print(f"Upload complete: {r.json()}")

    # 6. Мониторинг прогресса
    print("\n=== Pipeline Progress ===")
    while True:
        try:
            r = requests.get(f"{BASE_URL}/jobs/{job_id}/progress")
            r.raise_for_status()
            progress = r.json()
        except Exception as e:
            print(f"Error getting progress: {e}")
            time.sleep(5)
            continue

        status = progress.get("status", "unknown")
        overall = progress.get("overall_progress", 0)
        message = progress.get("message", "")
        
        line = f"[{overall:5.1f}%] {status}: {message}"
        
        training = progress.get("training")
        if training and training.get("current_step", 0) > 0:
            step = training["current_step"]
            total = training["total_steps"]
            psnr = training.get("psnr") or 0
            ssim = training.get("ssim") or 0
            splats = training.get("num_splats") or 0
            line += f" | Step {step}/{total} PSNR={psnr:.2f} SSIM={ssim:.4f} splats={splats}"
        
        print(line)

        if status in ("completed", "failed", "cancelled", "colmap_failed", "training_failed"):
            print(f"\n=== Final status: {status} ===")
            if status == "completed":
                print("Model ready! Use GET /jobs/{job_id}/model to download.")
            else:
                error = progress.get("error", "unknown error")
                print(f"Error: {error}")
            break
        
        time.sleep(5)

if __name__ == "__main__":
    main()
