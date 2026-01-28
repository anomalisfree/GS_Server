"""
Утилиты для работы с процессами и потоковым выводом
"""

import asyncio
import re
from typing import AsyncIterator, Callable, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ProcessOutput:
    """Вывод процесса"""
    line: str
    is_stderr: bool = False
    parsed: Dict[str, Any] = None
    
    
class OutputParser:
    """Базовый парсер вывода"""
    
    def parse(self, line: str) -> Dict[str, Any]:
        """Распарсить строку вывода"""
        return {}


class BrushOutputParser(OutputParser):
    """Парсер вывода Brush"""
    
    # Regex паттерны для разных форматов вывода
    PATTERNS = {
        # Основной прогресс: "Step 1000/30000" или "1000/30000"
        'step': re.compile(r'(?:Step|Iter(?:ation)?)?[:\s]*(\d+)\s*/\s*(\d+)', re.IGNORECASE),
        
        # PSNR: "PSNR: 25.5" или "PSNR=25.5" или "psnr: 25.5"
        'psnr': re.compile(r'PSNR[:\s=]+(\d+\.?\d*)', re.IGNORECASE),
        
        # SSIM: "SSIM: 0.85"
        'ssim': re.compile(r'SSIM[:\s=]+(\d+\.?\d*)', re.IGNORECASE),
        
        # Loss: "Loss: 0.0123" или "loss=0.0123"
        'loss': re.compile(r'(?:^|[\s|])loss[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)', re.IGNORECASE),
        
        # Splat count: "splats: 500000" или "500000 splats"
        'splats': re.compile(r'(?:splats?[:\s=]+(\d+)|(\d+)\s*splats?)', re.IGNORECASE),
        
        # Export: "Exported to model_5000.ply" или "Exporting iter 5000"
        'export': re.compile(r'export(?:ed|ing)?.*?(\d+)', re.IGNORECASE),
        
        # Learning rate: "lr: 0.0001" или "learning_rate=1e-4"
        'lr': re.compile(r'(?:lr|learning.?rate)[:\s=]+(\d+\.?\d*(?:e[+-]?\d+)?)', re.IGNORECASE),
        
        # Refine step: "Refined: 500000 splats"
        'refine': re.compile(r'refin(?:ed?|ing)[:\s]*(\d+)', re.IGNORECASE),
        
        # Speed: "10.5 steps/s" или "steps/sec: 10.5"
        'speed': re.compile(r'(\d+\.?\d*)\s*(?:steps?/?s(?:ec)?|it/?s)', re.IGNORECASE),
        
        # ETA: "ETA: 5m 30s" или "eta=330"
        'eta': re.compile(r'eta[:\s=]+(?:(\d+)m\s*)?(\d+)s?', re.IGNORECASE),
        
        # Memory: "GPU Memory: 5.2GB" или "vram: 5200MB"
        'memory': re.compile(r'(?:gpu\s*)?(?:memory|vram)[:\s=]+(\d+\.?\d*)\s*(GB|MB|KB)?', re.IGNORECASE),
        
        # Progress bar: "[=====>    ] 50%" или "50%"
        'percent': re.compile(r'(\d+(?:\.\d+)?)\s*%'),
        
        # Epoch: "Epoch 1/10"
        'epoch': re.compile(r'epoch[:\s]+(\d+)\s*/\s*(\d+)', re.IGNORECASE),
        
        # Time elapsed: "Elapsed: 1h 30m" или "time: 5400s"
        'elapsed': re.compile(r'(?:elapsed|time)[:\s=]+(?:(\d+)h\s*)?(?:(\d+)m\s*)?(\d+)s?', re.IGNORECASE),
    }
    
    def parse(self, line: str) -> Dict[str, Any]:
        """Распарсить строку вывода Brush"""
        result = {}
        
        for name, pattern in self.PATTERNS.items():
            match = pattern.search(line)
            if match:
                groups = match.groups()
                
                if name == 'step':
                    result['current_step'] = int(groups[0])
                    result['total_steps'] = int(groups[1])
                    
                elif name == 'psnr':
                    result['psnr'] = float(groups[0])
                    
                elif name == 'ssim':
                    result['ssim'] = float(groups[0])
                    
                elif name == 'loss':
                    result['loss'] = float(groups[0])
                    
                elif name == 'splats':
                    result['splat_count'] = int(groups[0] or groups[1])
                    
                elif name == 'export':
                    result['export_step'] = int(groups[0])
                    
                elif name == 'lr':
                    result['learning_rate'] = float(groups[0])
                    
                elif name == 'refine':
                    result['refine_count'] = int(groups[0])
                    
                elif name == 'speed':
                    result['steps_per_second'] = float(groups[0])
                    
                elif name == 'eta':
                    minutes = int(groups[0]) if groups[0] else 0
                    seconds = int(groups[1]) if groups[1] else 0
                    result['eta_seconds'] = minutes * 60 + seconds
                    
                elif name == 'memory':
                    value = float(groups[0])
                    unit = (groups[1] or 'MB').upper()
                    if unit == 'GB':
                        value *= 1024
                    elif unit == 'KB':
                        value /= 1024
                    result['memory_mb'] = value
                    
                elif name == 'percent':
                    result['percent'] = float(groups[0])
                    
                elif name == 'epoch':
                    result['epoch'] = int(groups[0])
                    result['total_epochs'] = int(groups[1])
                    
                elif name == 'elapsed':
                    hours = int(groups[0]) if groups[0] else 0
                    minutes = int(groups[1]) if groups[1] else 0
                    seconds = int(groups[2]) if groups[2] else 0
                    result['elapsed_seconds'] = hours * 3600 + minutes * 60 + seconds
        
        return result


class ColmapOutputParser(OutputParser):
    """Парсер вывода COLMAP"""
    
    PATTERNS = {
        # Processing image: "Processing image [5/100]"
        'image_processing': re.compile(r'Processing image \[(\d+)/(\d+)\]'),
        
        # Feature extraction: "Extracted 5000 features"
        'features': re.compile(r'Extracted (\d+) features'),
        
        # Matching: "Matching block [5/50]"
        'matching_block': re.compile(r'Matching block \[(\d+)/(\d+)'),
        
        # Matches found: "Found 1000 matches"
        'matches': re.compile(r'Found (\d+) matches'),
        
        # Registered images: "Registering image #5 (100)"
        'registered': re.compile(r'Registering image #\d+ \((\d+)\)'),
        
        # 3D points: "Points3D: 50000"
        'points3d': re.compile(r'Points3D:\s*(\d+)'),
        
        # Undistortion: "Undistorting image [5/100]"
        'undistortion': re.compile(r'Undistorting image \[(\d+)/(\d+)\]'),
        
        # Error messages
        'error': re.compile(r'(?:error|failed|exception)', re.IGNORECASE),
        
        # Warning messages
        'warning': re.compile(r'(?:warning|warn)', re.IGNORECASE),
    }
    
    def parse(self, line: str) -> Dict[str, Any]:
        """Распарсить строку вывода COLMAP"""
        result = {}
        
        for name, pattern in self.PATTERNS.items():
            match = pattern.search(line)
            if match:
                groups = match.groups()
                
                if name == 'image_processing':
                    result['images_processed'] = int(groups[0])
                    result['total_images'] = int(groups[1])
                    
                elif name == 'features':
                    result['features_extracted'] = int(groups[0])
                    
                elif name == 'matching_block':
                    result['matching_block'] = int(groups[0])
                    result['total_blocks'] = int(groups[1])
                    
                elif name == 'matches':
                    result['matches_found'] = int(groups[0])
                    
                elif name == 'registered':
                    result['registered_images'] = int(groups[0])
                    
                elif name == 'points3d':
                    result['points_3d'] = int(groups[0])
                    
                elif name == 'undistortion':
                    result['undistorted'] = int(groups[0])
                    result['total_undistort'] = int(groups[1])
                    
                elif name == 'error':
                    result['is_error'] = True
                    
                elif name == 'warning':
                    result['is_warning'] = True
        
        return result


async def stream_process_output(
    process: asyncio.subprocess.Process,
    parser: Optional[OutputParser] = None,
    on_line: Optional[Callable[[ProcessOutput], None]] = None,
    combine_stderr: bool = True
) -> AsyncIterator[ProcessOutput]:
    """
    Стримить вывод процесса построчно
    
    Args:
        process: asyncio subprocess
        parser: Парсер для анализа вывода
        on_line: Callback для каждой строки
        combine_stderr: Объединять stderr с stdout
        
    Yields:
        ProcessOutput для каждой строки
    """
    async def read_stream(stream, is_stderr: bool):
        async for line in stream:
            text = line.decode('utf-8', errors='ignore').strip()
            if not text:
                continue
                
            parsed = parser.parse(text) if parser else {}
            output = ProcessOutput(line=text, is_stderr=is_stderr, parsed=parsed)
            
            if on_line:
                on_line(output)
                
            yield output
    
    if combine_stderr and process.stderr:
        # Объединить потоки
        async for output in read_stream(process.stdout, False):
            yield output
    else:
        # Только stdout
        async for output in read_stream(process.stdout, False):
            yield output


def format_time(seconds: float) -> str:
    """Форматировать время в читаемый вид"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def format_size(bytes: int) -> str:
    """Форматировать размер файла"""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.1f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / 1024 / 1024:.1f} MB"
    else:
        return f"{bytes / 1024 / 1024 / 1024:.2f} GB"
