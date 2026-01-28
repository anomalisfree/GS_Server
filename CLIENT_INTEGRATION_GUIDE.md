# GS Server Client Integration Guide

## Client Application Integration with GS Server

**Server Address:** `http://192.168.1.14:8080`  
**WebSocket:** `ws://192.168.1.14:8080`

---

## Table of Contents

1. [API Overview](#api-overview)
2. [Python Client](#python-client)
3. [C# / Unity Client](#c-unity-client)
4. [JavaScript / Web Client](#javascript-web-client)
5. [cURL Examples](#curl-examples)
6. [Chunked Upload for Large Files](#chunked-upload)
7. [WebSocket Monitoring](#websocket-monitoring)
8. [Error Handling](#error-handling)

---

## API Overview

### Basic Workflow

```
1. POST /jobs              → Create a job (get job_id)
2. POST /jobs/{id}/upload  → Upload images
3. POST /jobs/{id}/upload/complete → Complete upload, start processing
4. GET /jobs/{id}/progress → Track progress (or WebSocket)
5. GET /jobs/{id}/model    → Download the trained model
```

### Job Statuses

| Status | Description |
|--------|-------------|
| `pending` | Waiting for upload |
| `uploading` | Upload in progress |
| `uploaded` | Upload completed |
| `colmap_running` | COLMAP processing |
| `colmap_done` | COLMAP completed |
| `training` | Brush model training |
| `completed` | Done! |
| `failed` | Error |

---

## Python Client

### Install Dependencies

```bash
pip install requests websockets aiohttp
```

### Complete Example

```python
"""
GS Server Python Client
Send images for Gaussian Splatting training
"""

import os
import time
import requests
from pathlib import Path
from typing import List, Optional, Callable
import asyncio
import json

class GSClient:
    """Client for GS Server"""
    
    def __init__(self, server_url: str = "http://192.168.1.14:8080"):
        self.base_url = server_url.rstrip('/')
        self.ws_url = server_url.replace('http://', 'ws://').replace('https://', 'wss://')
        self.timeout = 30
        
    # ==================== Core Methods ====================
    
    def check_connection(self) -> bool:
        """Check connection to server"""
        try:
            r = requests.get(f"{self.base_url}/status", timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def get_status(self) -> dict:
        """Get server status"""
        r = requests.get(f"{self.base_url}/status", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def create_job(
        self,
        name: str,
        auto_start: bool = True,
        total_steps: int = 30000,
        max_resolution: int = 1920
    ) -> dict:
        """
        Create a new job
        
        Args:
            name: Job name (scene name)
            auto_start: Automatically start processing after upload
            total_steps: Number of training steps
            max_resolution: Maximum image resolution
            
        Returns:
            dict with job_id and upload_url
        """
        payload = {
            "name": name,
            "auto_start": auto_start,
            "brush_config": {
                "total_steps": total_steps,
                "max_resolution": max_resolution
            }
        }
        
        r = requests.post(f"{self.base_url}/jobs", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def upload_images(
        self, 
        job_id: str, 
        image_paths: List[str],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> dict:
        """
        Upload images
        
        Args:
            job_id: Job ID
            image_paths: List of image paths
            on_progress: Callback(uploaded_count, total_count)
        """
        total = len(image_paths)
        uploaded = 0
        
        # Upload in batches of 20 files
        batch_size = 20
        
        for i in range(0, total, batch_size):
            batch = image_paths[i:i + batch_size]
            files = []
            
            for path in batch:
                filename = os.path.basename(path)
                files.append(('files', (filename, open(path, 'rb'), 'image/jpeg')))
            
            try:
                r = requests.post(
                    f"{self.base_url}/jobs/{job_id}/upload",
                    files=files,
                    timeout=300  # 5 minutes per batch
                )
                r.raise_for_status()
            finally:
                # Close files
                for _, (_, f, _) in files:
                    f.close()
            
            uploaded += len(batch)
            if on_progress:
                on_progress(uploaded, total)
        
        return {"uploaded": uploaded, "total": total}
    
    def upload_directory(
        self,
        job_id: str,
        directory: str,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> dict:
        """
        Upload all images from a directory
        
        Args:
            job_id: Job ID
            directory: Path to directory with images
            extensions: File extensions to upload
            on_progress: Progress callback
        """
        dir_path = Path(directory)
        images = []
        
        for ext in extensions:
            images.extend(dir_path.glob(f'*{ext}'))
            images.extend(dir_path.glob(f'*{ext.upper()}'))
        
        image_paths = sorted([str(p) for p in images])
        
        if not image_paths:
            raise ValueError(f"No images found in {directory}")
        
        print(f"Found {len(image_paths)} images")
        return self.upload_images(job_id, image_paths, on_progress)
    
    def complete_upload(self, job_id: str) -> dict:
        """Complete upload and start processing"""
        r = requests.post(
            f"{self.base_url}/jobs/{job_id}/upload/complete",
            timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()
    
    def get_progress(self, job_id: str) -> dict:
        """Get current progress"""
        r = requests.get(
            f"{self.base_url}/jobs/{job_id}/progress",
            timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        on_progress: Optional[Callable[[dict], None]] = None
    ) -> dict:
        """
        Wait for job completion (polling)
        
        Args:
            job_id: Job ID
            poll_interval: Polling interval in seconds
            on_progress: Callback for progress updates
        """
        terminal_statuses = {'completed', 'failed', 'cancelled', 
                            'colmap_failed', 'training_failed'}
        
        while True:
            progress = self.get_progress(job_id)
            
            if on_progress:
                on_progress(progress)
            
            if progress['status'] in terminal_statuses:
                return progress
            
            time.sleep(poll_interval)
    
    def download_model(self, job_id: str, output_path: str) -> str:
        """
        Download the trained model
        
        Args:
            job_id: Job ID
            output_path: Path to save .ply file
        """
        r = requests.get(
            f"{self.base_url}/jobs/{job_id}/model",
            stream=True,
            timeout=600  # 10 minutes for download
        )
        r.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
    
    def get_job_info(self, job_id: str) -> dict:
        """Get full job information"""
        r = requests.get(f"{self.base_url}/jobs/{job_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def cancel_job(self, job_id: str) -> dict:
        """Cancel a job"""
        r = requests.post(f"{self.base_url}/jobs/{job_id}/cancel", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    def delete_job(self, job_id: str) -> dict:
        """Delete a job"""
        r = requests.delete(f"{self.base_url}/jobs/{job_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
    
    # ==================== Chunked Upload ====================
    
    def upload_large_file(
        self,
        job_id: str,
        filepath: str,
        chunk_size: int = 10 * 1024 * 1024,  # 10 MB
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> dict:
        """
        Upload a large file in chunks (for files > 100MB)
        
        Args:
            job_id: Job ID
            filepath: Path to file
            chunk_size: Chunk size in bytes
            on_progress: Callback(uploaded_bytes, total_bytes)
        """
        import math
        
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        total_chunks = math.ceil(file_size / chunk_size)
        
        with open(filepath, 'rb') as f:
            for i in range(total_chunks):
                chunk_data = f.read(chunk_size)
                
                r = requests.post(
                    f"{self.base_url}/jobs/{job_id}/upload/chunk",
                    data={
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": total_chunks
                    },
                    files={"chunk": chunk_data},
                    timeout=120
                )
                r.raise_for_status()
                
                if on_progress:
                    uploaded = min((i + 1) * chunk_size, file_size)
                    on_progress(uploaded, file_size)
        
        return {"status": "complete", "filename": filename}


# ==================== Usage Example ====================

def main():
    """Complete workflow example"""
    
    # Initialize client
    client = GSClient("http://192.168.1.14:8080")
    
    # Check connection
    if not client.check_connection():
        print("❌ Server unavailable!")
        return
    
    print("✓ Connected to server")
    status = client.get_status()
    print(f"  Active jobs: {status['active_jobs']}")
    print(f"  Queued jobs: {status['queued_jobs']}")
    
    # Path to images folder
    images_folder = r"C:\MyImages\scene1"  # Change to your path
    
    # 1. Create job
    print("\n📝 Creating job...")
    job = client.create_job(
        name="my_scene",
        total_steps=30000,      # Number of steps (more = better quality)
        max_resolution=1920     # Max resolution
    )
    job_id = job['job_id']
    print(f"✓ Job created: {job_id}")
    
    # 2. Upload images
    print(f"\n📤 Uploading images from {images_folder}...")
    
    def on_upload_progress(uploaded, total):
        percent = (uploaded / total) * 100
        print(f"\r  Uploaded: {uploaded}/{total} ({percent:.1f}%)", end='', flush=True)
    
    try:
        client.upload_directory(job_id, images_folder, on_progress=on_upload_progress)
        print("\n✓ Upload completed")
    except Exception as e:
        print(f"\n❌ Upload error: {e}")
        return
    
    # 3. Complete upload and start processing
    print("\n🚀 Starting processing...")
    client.complete_upload(job_id)
    
    # 4. Track progress
    print("\n⏳ Waiting for completion...\n")
    
    def on_progress(progress):
        status = progress['status']
        overall = progress['overall_progress']
        msg = progress.get('message', '')
        
        print(f"\r[{overall:5.1f}%] {status}: {msg}", end='', flush=True)
        
        # Training details
        if progress.get('training'):
            t = progress['training']
            step = t.get('current_step', 0)
            total = t.get('total_steps', 0)
            psnr = t.get('psnr')
            
            extra = f" | Step {step}/{total}"
            if psnr:
                extra += f" | PSNR: {psnr:.2f}"
            print(extra, end='', flush=True)
        
        print("          ", end='')  # Clear rest of line
    
    result = client.wait_for_completion(job_id, poll_interval=3.0, on_progress=on_progress)
    print()
    
    # 5. Download result
    if result['status'] == 'completed':
        print("\n✅ Training completed!")
        
        output_file = f"{job['name']}_model.ply"
        print(f"\n📥 Downloading model to {output_file}...")
        client.download_model(job_id, output_file)
        
        # Model info
        info = client.get_job_info(job_id)
        print(f"\n📊 Statistics:")
        print(f"  Images: {info.get('images_count', 'N/A')}")
        
        if info.get('progress', {}).get('training'):
            t = info['progress']['training']
            print(f"  Splats: {t.get('splat_count', 'N/A'):,}")
            if t.get('psnr'):
                print(f"  PSNR: {t['psnr']:.2f}")
            if t.get('ssim'):
                print(f"  SSIM: {t['ssim']:.4f}")
        
        print(f"\n✓ Model saved: {output_file}")
    else:
        print(f"\n❌ Error: {result['status']}")
        if result.get('error'):
            print(f"   {result['error']}")


if __name__ == '__main__':
    main()
```

### Async Client with WebSocket

```python
"""
Async client with WebSocket monitoring
"""

import asyncio
import aiohttp
import json
from typing import Callable, Optional


class AsyncGSClient:
    """Async client for GS Server"""
    
    def __init__(self, server_url: str = "http://192.168.1.14:8080"):
        self.base_url = server_url.rstrip('/')
        self.ws_url = server_url.replace('http://', 'ws://')
        
    async def create_job(self, name: str, **config) -> dict:
        """Create a job"""
        async with aiohttp.ClientSession() as session:
            payload = {"name": name, "auto_start": True}
            if config:
                payload["brush_config"] = config
                
            async with session.post(f"{self.base_url}/jobs", json=payload) as r:
                return await r.json()
    
    async def upload_images(self, job_id: str, image_paths: list) -> dict:
        """Upload images"""
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            
            for path in image_paths:
                data.add_field('files',
                              open(path, 'rb'),
                              filename=path.split('/')[-1].split('\\')[-1])
            
            async with session.post(
                f"{self.base_url}/jobs/{job_id}/upload",
                data=data
            ) as r:
                return await r.json()
    
    async def complete_upload(self, job_id: str) -> dict:
        """Complete upload"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/jobs/{job_id}/upload/complete"
            ) as r:
                return await r.json()
    
    async def monitor_progress(
        self,
        job_id: str,
        on_message: Callable[[dict], None]
    ):
        """
        Monitor progress via WebSocket
        
        Args:
            job_id: Job ID
            on_message: Callback for each message
        """
        uri = f"{self.ws_url}/jobs/{job_id}/ws"
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(uri) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        on_message(data)
                        
                        # End on completion
                        if data['type'] in ['completed', 'error']:
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
    
    async def download_model(self, job_id: str, output_path: str):
        """Download model"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/jobs/{job_id}/model") as r:
                with open(output_path, 'wb') as f:
                    async for chunk in r.content.iter_chunked(8192):
                        f.write(chunk)


async def main_async():
    """Async example with WebSocket"""
    
    client = AsyncGSClient("http://192.168.1.14:8080")
    
    # Create job
    job = await client.create_job("my_scene", total_steps=30000)
    job_id = job['job_id']
    print(f"Job created: {job_id}")
    
    # Upload images (replace with your paths)
    images = ["image1.jpg", "image2.jpg"]  # Your images
    await client.upload_images(job_id, images)
    await client.complete_upload(job_id)
    
    # Monitor via WebSocket
    def on_message(data):
        if data['type'] == 'progress':
            p = data['data']
            print(f"[{p['overall_progress']:.1f}%] {p['status']}: {p['message']}")
        elif data['type'] == 'completed':
            print("✅ Completed!")
        elif data['type'] == 'error':
            print(f"❌ Error: {data['data'].get('error')}")
    
    await client.monitor_progress(job_id, on_message)
    
    # Download model
    await client.download_model(job_id, "model.ply")
    print("Model downloaded!")


if __name__ == '__main__':
    asyncio.run(main_async())
```

---

## C# / Unity Client

### Installation

For Unity, add the `Newtonsoft.Json` package via Package Manager.

### GSClient.cs

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

/// <summary>
/// Client for GS Server - Gaussian Splatting training
/// </summary>
public class GSClient
{
    private readonly string _baseUrl;
    private readonly HttpClient _httpClient;
    
    public GSClient(string serverUrl = "http://192.168.1.14:8080")
    {
        _baseUrl = serverUrl.TrimEnd('/');
        _httpClient = new HttpClient();
        _httpClient.Timeout = TimeSpan.FromMinutes(10);
    }
    
    #region Data Models
    
    [Serializable]
    public class CreateJobRequest
    {
        public string name;
        public bool auto_start = true;
        public Dictionary<string, object> brush_config;
    }
    
    [Serializable]
    public class JobResponse
    {
        public string job_id;
        public string upload_url;
        public string websocket_url;
    }
    
    [Serializable]
    public class ProgressResponse
    {
        public float overall_progress;
        public string status;
        public string message;
        public string error;
        public TrainingProgress training;
    }
    
    [Serializable]
    public class TrainingProgress
    {
        public int current_step;
        public int total_steps;
        public float steps_per_second;
        public float? psnr;
        public float? ssim;
        public int splat_count;
    }
    
    [Serializable]
    public class ServerStatus
    {
        public string status;
        public string version;
        public int active_jobs;
        public int queued_jobs;
    }
    
    #endregion
    
    #region API Methods
    
    /// <summary>
    /// Check connection to server
    /// </summary>
    public async Task<bool> CheckConnectionAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/status");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
    
    /// <summary>
    /// Get server status
    /// </summary>
    public async Task<ServerStatus> GetStatusAsync()
    {
        var response = await _httpClient.GetAsync($"{_baseUrl}/status");
        response.EnsureSuccessStatusCode();
        var json = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<ServerStatus>(json);
    }
    
    /// <summary>
    /// Create a new job
    /// </summary>
    public async Task<JobResponse> CreateJobAsync(
        string name, 
        int totalSteps = 30000,
        int maxResolution = 1920)
    {
        var request = new CreateJobRequest
        {
            name = name,
            auto_start = true,
            brush_config = new Dictionary<string, object>
            {
                { "total_steps", totalSteps },
                { "max_resolution", maxResolution }
            }
        };
        
        var json = JsonConvert.SerializeObject(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        var response = await _httpClient.PostAsync($"{_baseUrl}/jobs", content);
        response.EnsureSuccessStatusCode();
        
        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<JobResponse>(responseJson);
    }
    
    /// <summary>
    /// Upload images
    /// </summary>
    public async Task<JObject> UploadImagesAsync(
        string jobId, 
        string[] imagePaths,
        IProgress<float> progress = null)
    {
        int batchSize = 10;
        int totalUploaded = 0;
        
        for (int i = 0; i < imagePaths.Length; i += batchSize)
        {
            var batch = new List<string>();
            for (int j = i; j < Math.Min(i + batchSize, imagePaths.Length); j++)
            {
                batch.Add(imagePaths[j]);
            }
            
            using var formData = new MultipartFormDataContent();
            
            foreach (var path in batch)
            {
                var fileName = Path.GetFileName(path);
                var fileContent = new ByteArrayContent(File.ReadAllBytes(path));
                formData.Add(fileContent, "files", fileName);
            }
            
            var response = await _httpClient.PostAsync(
                $"{_baseUrl}/jobs/{jobId}/upload", 
                formData
            );
            response.EnsureSuccessStatusCode();
            
            totalUploaded += batch.Count;
            progress?.Report((float)totalUploaded / imagePaths.Length);
        }
        
        return new JObject { { "uploaded", totalUploaded } };
    }
    
    /// <summary>
    /// Complete upload and start processing
    /// </summary>
    public async Task CompleteUploadAsync(string jobId)
    {
        var response = await _httpClient.PostAsync(
            $"{_baseUrl}/jobs/{jobId}/upload/complete", 
            null
        );
        response.EnsureSuccessStatusCode();
    }
    
    /// <summary>
    /// Get job progress
    /// </summary>
    public async Task<ProgressResponse> GetProgressAsync(string jobId)
    {
        var response = await _httpClient.GetAsync($"{_baseUrl}/jobs/{jobId}/progress");
        response.EnsureSuccessStatusCode();
        var json = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<ProgressResponse>(json);
    }
    
    /// <summary>
    /// Wait for job completion
    /// </summary>
    public async Task<ProgressResponse> WaitForCompletionAsync(
        string jobId,
        float pollIntervalSeconds = 5f,
        Action<ProgressResponse> onProgress = null)
    {
        var terminalStatuses = new HashSet<string> 
        { 
            "completed", "failed", "cancelled", 
            "colmap_failed", "training_failed" 
        };
        
        while (true)
        {
            var progress = await GetProgressAsync(jobId);
            onProgress?.Invoke(progress);
            
            if (terminalStatuses.Contains(progress.status))
                return progress;
            
            await Task.Delay(TimeSpan.FromSeconds(pollIntervalSeconds));
        }
    }
    
    /// <summary>
    /// Download trained model
    /// </summary>
    public async Task DownloadModelAsync(string jobId, string outputPath)
    {
        var response = await _httpClient.GetAsync($"{_baseUrl}/jobs/{jobId}/model");
        response.EnsureSuccessStatusCode();
        
        using var stream = await response.Content.ReadAsStreamAsync();
        using var fileStream = File.Create(outputPath);
        await stream.CopyToAsync(fileStream);
    }
    
    /// <summary>
    /// Cancel a job
    /// </summary>
    public async Task CancelJobAsync(string jobId)
    {
        var response = await _httpClient.PostAsync(
            $"{_baseUrl}/jobs/{jobId}/cancel", 
            null
        );
        response.EnsureSuccessStatusCode();
    }
    
    #endregion
}
```

---

## JavaScript / Web Client

### Basic Client

```javascript
/**
 * GS Server JavaScript Client
 * For browser or Node.js
 */

class GSClient {
    constructor(serverUrl = 'http://192.168.1.14:8080') {
        this.baseUrl = serverUrl.replace(/\/$/, '');
        this.wsUrl = this.baseUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    }

    async getStatus() {
        const response = await fetch(`${this.baseUrl}/status`);
        return response.json();
    }

    async createJob(name, options = {}) {
        const payload = {
            name,
            auto_start: options.autoStart ?? true,
            brush_config: {
                total_steps: options.totalSteps ?? 30000,
                max_resolution: options.maxResolution ?? 1920
            }
        };

        const response = await fetch(`${this.baseUrl}/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return response.json();
    }

    async uploadImages(jobId, files, onProgress = null) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        const xhr = new XMLHttpRequest();
        
        return new Promise((resolve, reject) => {
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable && onProgress) {
                    onProgress(e.loaded, e.total);
                }
            };

            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(`Upload failed: ${xhr.status}`));
                }
            };

            xhr.onerror = () => reject(new Error('Network error'));

            xhr.open('POST', `${this.baseUrl}/jobs/${jobId}/upload`);
            xhr.send(formData);
        });
    }

    async completeUpload(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}/upload/complete`, {
            method: 'POST'
        });
        return response.json();
    }

    async getProgress(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}/progress`);
        return response.json();
    }

    async cancelJob(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}/cancel`, {
            method: 'POST'
        });
        return response.json();
    }

    async downloadModel(jobId) {
        const response = await fetch(`${this.baseUrl}/jobs/${jobId}/model`);
        return response.blob();
    }

    monitorProgress(jobId, callbacks = {}) {
        const { onProgress, onComplete, onError, onMessage } = callbacks;
        
        const ws = new WebSocket(`${this.wsUrl}/jobs/${jobId}/ws`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage?.(data);

            switch (data.type) {
                case 'progress':
                    onProgress?.(data.data);
                    break;
                case 'completed':
                    onComplete?.(data.data);
                    ws.close();
                    break;
                case 'error':
                    onError?.(data.data.error);
                    ws.close();
                    break;
            }
        };

        return { close: () => ws.close(), ws };
    }
}
```

---

## cURL Examples

### Check Server

```bash
curl http://192.168.1.14:8080/status
```

### Create Job

```bash
curl -X POST http://192.168.1.14:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{"name": "my_scene", "auto_start": true, "brush_config": {"total_steps": 30000}}'
```

### Upload Images

```bash
curl -X POST http://192.168.1.14:8080/jobs/JOB_ID/upload \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Complete Upload

```bash
curl -X POST http://192.168.1.14:8080/jobs/JOB_ID/upload/complete
```

### Get Progress

```bash
curl http://192.168.1.14:8080/jobs/JOB_ID/progress
```

### Download Model

```bash
curl -o model.ply http://192.168.1.14:8080/jobs/JOB_ID/model
```

---

## WebSocket Monitoring

### Message Format

```javascript
// Progress update
{
    "type": "progress",
    "job_id": "abc123...",
    "data": {
        "overall_progress": 45.5,
        "status": "training",
        "message": "Step 5000/30000",
        "training": {
            "current_step": 5000,
            "total_steps": 30000,
            "psnr": 25.3,
            "ssim": 0.85
        }
    }
}

// Completed
{
    "type": "completed",
    "job_id": "abc123...",
    "data": { "model_path": "/path/to/model.ply" }
}

// Error
{
    "type": "error",
    "job_id": "abc123...",
    "data": { "error": "COLMAP failed" }
}
```

---

## Error Handling

### HTTP Codes

| Code | Description |
|------|-------------|
| 200 | OK |
| 400 | Bad Request |
| 404 | Job not found |
| 500 | Server error |

---

## Recommendations

1. **Number of images**: 50-500 for best results
2. **Resolution**: 1080p-4K recommended
3. **Quality**: Avoid blurry photos, good lighting
4. **Coverage**: Capture from all sides with 60-80% overlap
5. **Format**: JPG/PNG, RGB (not RGBA)

---

## Support

- API Docs: http://192.168.1.14:8080/docs
- Status: http://192.168.1.14:8080/status
