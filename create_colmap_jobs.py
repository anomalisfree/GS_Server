"""
Create colmap_images jobs for all 12 folders, uploading images via the real
POST /jobs/{id}/upload + /upload/complete API (not SSH), in batches.
"""
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://136.64.198.146:8080"
DATA_PREPARED = Path(r"D:\VSProjects\GS_Sever\data_prepared")
BATCH_SIZE = 40

FOLDER_MAP = [
    ("C_Deck_FWD", "C_Deck_FWD"),
    ("D_Deck_FWD_Section 1 (Front Tip)", "D_Deck_FWD_Section_1_Front_Tip"),
    ("D_Deck_FWD_Section 2(behind section 1)", "D_Deck_FWD_Section_2_behind_section_1"),
    ("SSUS_AFT_Engine_Room_ES22_iPhone_P3DGL", "SSUS_AFT_Engine_Room_ES22_iPhone_P3DGL"),
    ("SSUS_AFT_Engine_Room_P2.1_02_iPad_P3DNW", "SSUS_AFT_Engine_Room_P2_1_02_iPad_P3DNW"),
    ("TY_UpperDeckAFT_UseThisOne", "TY_UpperDeckAFT_UseThisOne"),
    # core_machines_next_to_long_tube excluded: registered 192/192 (100%) on the first
    # attempt already, no need to re-run.
    ("core propellers 2", "core_propellers_2"),
    ("engine room b deck", "engine_room_b_deck"),
    ("front tip full", "front_tip_full"),
    ("inside big towers", "inside_big_towers"),
    ("long silver tube", "long_silver_tube"),
]

SEQUENTIAL_OVERLAP = 50  # was fragmenting badly at COLMAP's default of 10
JOB_SUFFIX = "_colmap_v2"


def create_and_upload(local_name: str, server_name: str) -> str:
    images_dir = DATA_PREPARED / local_name / "images"
    images = sorted(images_dir.glob("*.jpg"))
    print(f"\n{'='*70}\n{server_name}  ({len(images)} images)\n{'='*70}")

    resp = requests.post(f"{BASE_URL}/jobs", json={
        "name": f"{server_name}{JOB_SUFFIX}",
        "training_mode": "colmap_images",
        "auto_start": True,
        "colmap_config": {
            "matcher_type": "sequential",
            "sequential_loop_detection": True,
            "sequential_overlap": SEQUENTIAL_OVERLAP,
            "camera_model": "OPENCV",
        },
        "brush_config": {"total_steps": 30000, "max_resolution": 1024},
    })
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"  job_id={job_id}")

    uploaded = 0
    for i in range(0, len(images), BATCH_SIZE):
        batch = images[i:i + BATCH_SIZE]
        files = [("files", (p.name, open(p, "rb"), "image/jpeg")) for p in batch]
        try:
            r = requests.post(f"{BASE_URL}/jobs/{job_id}/upload", files=files, timeout=120)
            r.raise_for_status()
        finally:
            for _, (_, fh, _) in files:
                fh.close()
        uploaded += len(batch)
        print(f"  uploaded {uploaded}/{len(images)}", end="\r")
    print()

    r = requests.post(f"{BASE_URL}/jobs/{job_id}/upload/complete")
    r.raise_for_status()
    print(f"  upload/complete: {r.json()}")
    return job_id


def main():
    results = []
    for local_name, server_name in FOLDER_MAP:
        for attempt in range(3):
            try:
                job_id = create_and_upload(local_name, server_name)
                results.append((local_name, server_name, job_id))
                break
            except Exception as e:
                print(f"  [ERROR attempt {attempt+1}] {e}")
                time.sleep(5)
        else:
            print(f"  [FAILED] giving up on {server_name}")
            results.append((local_name, server_name, "FAILED"))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    with open("colmap_jobs_v2_created.txt", "w") as f:
        for local_name, server_name, job_id in results:
            line = f"{local_name}|{server_name}|colmap|{job_id}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
