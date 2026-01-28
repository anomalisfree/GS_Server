"""
Пример Python клиента для GS Server
"""

import requests
import asyncio
import websockets
import json
import time
import os
import math
from pathlib import Path
from typing import List, Optional


class GSClient:
    """Клиент для работы с GS Server"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.ws_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        
    def get_status(self) -> dict:
        """Получить статус сервера"""
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def create_job(
        self, 
        name: str, 
        auto_start: bool = True,
        colmap_config: Optional[dict] = None,
        brush_config: Optional[dict] = None
    ) -> dict:
        """Создать новую задачу"""
        payload = {
            "name": name,
            "auto_start": auto_start,
        }
        if colmap_config:
            payload["colmap_config"] = colmap_config
        if brush_config:
            payload["brush_config"] = brush_config
            
        response = requests.post(f"{self.base_url}/jobs", json=payload)
        response.raise_for_status()
        return response.json()
    
    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> dict:
        """Получить список задач"""
        params = {"limit": limit}
        if status:
            params["status"] = status
        response = requests.get(f"{self.base_url}/jobs", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_job(self, job_id: str) -> dict:
        """Получить информацию о задаче"""
        response = requests.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_progress(self, job_id: str) -> dict:
        """Получить прогресс задачи"""
        response = requests.get(f"{self.base_url}/jobs/{job_id}/progress")
        response.raise_for_status()
        return response.json()
    
    def upload_images(self, job_id: str, image_paths: List[str]) -> dict:
        """Загрузить изображения"""
        files = []
        for path in image_paths:
            files.append(("files", (os.path.basename(path), open(path, "rb"))))
        
        try:
            response = requests.post(f"{self.base_url}/jobs/{job_id}/upload", files=files)
            response.raise_for_status()
            return response.json()
        finally:
            for _, (_, f) in files:
                f.close()
    
    def upload_directory(self, job_id: str, directory: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> dict:
        """Загрузить все изображения из директории"""
        dir_path = Path(directory)
        images = []
        
        for ext in extensions:
            images.extend(dir_path.glob(f'*{ext}'))
            images.extend(dir_path.glob(f'*{ext.upper()}'))
        
        images = [str(p) for p in images]
        print(f"Found {len(images)} images in {directory}")
        
        # Загружаем пачками по 50 файлов
        batch_size = 50
        total_uploaded = 0
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            result = self.upload_images(job_id, batch)
            total_uploaded += result.get('uploaded', 0)
            print(f"Uploaded {total_uploaded}/{len(images)} images")
        
        return {"total_uploaded": total_uploaded}
    
    def upload_large_file(
        self, 
        job_id: str, 
        filepath: str, 
        chunk_size: int = 10 * 1024 * 1024
    ) -> dict:
        """Загрузить большой файл частями (chunked upload)"""
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        total_chunks = math.ceil(file_size / chunk_size)
        
        with open(filepath, 'rb') as f:
            for i in range(total_chunks):
                chunk_data = f.read(chunk_size)
                
                response = requests.post(
                    f"{self.base_url}/jobs/{job_id}/upload/chunk",
                    data={
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": total_chunks
                    },
                    files={"chunk": chunk_data}
                )
                response.raise_for_status()
                
                print(f"Uploaded chunk {i + 1}/{total_chunks} for {filename}")
        
        return {"status": "complete", "filename": filename}
    
    def complete_upload(self, job_id: str) -> dict:
        """Завершить загрузку и запустить обработку"""
        response = requests.post(f"{self.base_url}/jobs/{job_id}/upload/complete")
        response.raise_for_status()
        return response.json()
    
    def start_job(self, job_id: str, brush_config: Optional[dict] = None) -> dict:
        """Запустить задачу вручную"""
        payload = {}
        if brush_config:
            payload["brush_config"] = brush_config
            
        response = requests.post(f"{self.base_url}/jobs/{job_id}/start", json=payload or None)
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id: str) -> dict:
        """Отменить задачу"""
        response = requests.post(f"{self.base_url}/jobs/{job_id}/cancel")
        response.raise_for_status()
        return response.json()
    
    def delete_job(self, job_id: str) -> dict:
        """Удалить задачу"""
        response = requests.delete(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def download_model(self, job_id: str, output_path: str) -> str:
        """Скачать готовую модель"""
        response = requests.get(f"{self.base_url}/jobs/{job_id}/model", stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
    
    def get_model_info(self, job_id: str) -> dict:
        """Получить информацию о модели"""
        response = requests.get(f"{self.base_url}/jobs/{job_id}/model/info")
        response.raise_for_status()
        return response.json()
    
    def list_exports(self, job_id: str) -> dict:
        """Получить список всех экспортов"""
        response = requests.get(f"{self.base_url}/jobs/{job_id}/exports")
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(
        self, 
        job_id: str, 
        poll_interval: float = 5.0,
        callback=None
    ) -> dict:
        """Ждать завершения задачи (polling)"""
        while True:
            progress = self.get_progress(job_id)
            status = progress['status']
            
            if callback:
                callback(progress)
            else:
                print(f"Status: {status}, Progress: {progress['overall_progress']:.1f}%")
                
                if progress.get('training'):
                    t = progress['training']
                    print(f"  Training: {t['current_step']}/{t['total_steps']}")
                    if t.get('psnr'):
                        print(f"  PSNR: {t['psnr']:.2f}, SSIM: {t.get('ssim', 0):.4f}")
            
            if status in ['completed', 'failed', 'cancelled', 'colmap_failed', 'training_failed']:
                return progress
            
            time.sleep(poll_interval)
    
    async def monitor_progress(self, job_id: str, callback=None):
        """Мониторинг прогресса через WebSocket"""
        uri = f"{self.ws_url}/jobs/{job_id}/ws"
        
        async with websockets.connect(uri) as ws:
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    data = json.loads(message)
                    
                    if callback:
                        should_continue = callback(data)
                        if should_continue is False:
                            break
                    else:
                        self._default_ws_handler(data)
                    
                    if data['type'] in ['completed', 'error']:
                        break
                        
                except asyncio.TimeoutError:
                    # Отправить ping
                    await ws.send("ping")
    
    def _default_ws_handler(self, data: dict):
        """Обработчик WebSocket сообщений по умолчанию"""
        msg_type = data['type']
        
        if msg_type == 'initial':
            print(f"Connected to job: {data['job_id']}")
            print(f"Status: {data['data']['status']}")
            
        elif msg_type == 'progress':
            progress = data['data']
            print(f"\rProgress: {progress['overall_progress']:.1f}% - {progress['message']}", end='')
            
            if progress.get('training'):
                t = progress['training']
                if t['current_step'] > 0:
                    print(f" | Step {t['current_step']}/{t['total_steps']}", end='')
                    if t.get('psnr'):
                        print(f" | PSNR: {t['psnr']:.2f}", end='')
            print()
            
        elif msg_type == 'completed':
            print(f"\n✓ Job completed! Model: {data['data'].get('model_path')}")
            
        elif msg_type == 'error':
            print(f"\n✗ Error: {data['data'].get('error')}")


def main():
    """Пример использования клиента"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GS Server Client')
    parser.add_argument('--server', default='http://localhost:8080', help='Server URL')
    parser.add_argument('--images', help='Path to images directory')
    parser.add_argument('--name', default='my_scene', help='Job name')
    parser.add_argument('--steps', type=int, default=30000, help='Training steps')
    parser.add_argument('--output', default='model.ply', help='Output path for model')
    
    args = parser.parse_args()
    
    client = GSClient(args.server)
    
    # Проверить статус сервера
    print("Connecting to server...")
    status = client.get_status()
    print(f"Server status: {status['status']}")
    print(f"Active jobs: {status['active_jobs']}, Queued: {status['queued_jobs']}")
    
    if not args.images:
        print("\nUsage: python client_example.py --images <path_to_images>")
        return
    
    # Создать задачу
    print(f"\nCreating job '{args.name}'...")
    job = client.create_job(
        name=args.name,
        brush_config={"total_steps": args.steps}
    )
    job_id = job['job_id']
    print(f"Job created: {job_id}")
    
    # Загрузить изображения
    print(f"\nUploading images from {args.images}...")
    client.upload_directory(job_id, args.images)
    
    # Завершить загрузку
    print("\nCompleting upload and starting processing...")
    client.complete_upload(job_id)
    
    # Ждать завершения
    print("\nWaiting for completion...")
    result = client.wait_for_completion(job_id)
    
    if result['status'] == 'completed':
        print(f"\n✓ Training completed!")
        
        # Скачать модель
        print(f"Downloading model to {args.output}...")
        client.download_model(job_id, args.output)
        print(f"✓ Model saved to {args.output}")
        
        # Показать информацию о модели
        info = client.get_model_info(job_id)
        print(f"\nModel info:")
        print(f"  Size: {info['model_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Splats: {info['splat_count']:,}")
        if info.get('final_psnr'):
            print(f"  PSNR: {info['final_psnr']:.2f}")
    else:
        print(f"\n✗ Job failed with status: {result['status']}")
        if result.get('error'):
            print(f"Error: {result['error']}")


if __name__ == '__main__':
    main()
