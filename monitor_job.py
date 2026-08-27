"""Monitor existing job progress"""
import requests, time, sys

BASE_URL = "http://localhost:8080"
JOB_ID = "ded4dc10-1c67-40fb-a760-c3487b4a2a47"

print(f"Monitoring job {JOB_ID}...")
while True:
    try:
        r = requests.get(f"{BASE_URL}/jobs/{JOB_ID}/progress", timeout=5)
        r.raise_for_status()
        p = r.json()
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
        continue

    status = p.get("status", "?")
    overall = p.get("overall_progress", 0)
    msg = p.get("message", "")
    line = f"[{overall:5.1f}%] {status}: {msg}"

    t = p.get("training")
    if t and t.get("current_step", 0) > 0:
        psnr = t.get("psnr") or 0
        ssim = t.get("ssim") or 0
        line += f" | Step {t['current_step']}/{t['total_steps']} PSNR={psnr:.2f}"

    print(line)

    if status in ("completed", "failed", "cancelled", "colmap_failed", "training_failed"):
        print(f"\nFinal: {status}")
        if p.get("error"):
            print(f"Error: {p['error']}")
        break

    time.sleep(5)
