"""
Mask Generator - semantic segmentation via DeepLabV3 for background/sky removal
Generates masks that COLMAP uses to ignore unwanted regions during feature extraction.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set

from .models import JobStatus
from .config import get_config

if TYPE_CHECKING:
    from .job_manager import JobManager, Job

logger = logging.getLogger(__name__)

# Classes to mask out (COCO/VOC label indices for DeepLabV3)
# 0=background is NOT masked — it's "unknown" and usually fine
MASKABLE_CLASSES = {
    "sky": 0,       # not VOC — handled separately via custom index
    "person": 15,
    "car": 7,
    "bus": 6,
    "truck": 14,    # not in VOC21 but mapped nearby
    "motorcycle": 14,
    "bicycle": 2,
    "bird": 3,
    "cat": 8,
    "dog": 12,
    "horse": 13,
    "sheep": 17,
    "cow": 10,
    "airplane": 1,  # aeroplane
    "boat": 4,
    "train": 19,
}

# Default: remove sky and people (most impactful for 3D reconstruction)
DEFAULT_REMOVE_CLASSES = {"sky", "person"}


def _get_deeplabv3_class_index(class_name: str) -> Optional[int]:
    """Map human-readable class name to DeepLabV3 (VOC) label index.
    
    VOC class indices:
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle,
    6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog,
    13=horse, 14=motorbike, 15=person, 16=pottedplant, 17=sheep,
    18=sofa, 19=train, 20=tvmonitor
    """
    voc_map = {
        "background": 0,
        "aeroplane": 1, "airplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14, "motorcycle": 14,
        "person": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20, "tv": 20,
    }
    return voc_map.get(class_name.lower())


class MaskGenerator:
    """Generates segmentation masks using DeepLabV3 to filter unwanted regions."""

    def __init__(self, job_manager: "JobManager"):
        self._job_manager = job_manager
        self._config = get_config()
        self._model = None
        self._device = None

    def _load_model(self):
        """Lazy-load the DeepLabV3 model."""
        if self._model is not None:
            return

        import torch
        from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading DeepLabV3 on {self._device}")

        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self._model = deeplabv3_resnet101(weights=weights).to(self._device)
        self._model.eval()
        self._transforms = weights.transforms()

        logger.info("DeepLabV3 model loaded")

    async def run(self, job: "Job", remove_classes: Optional[Set[str]] = None):
        """Generate masks for all images in the job.

        COLMAP expects masks in a parallel directory structure:
        images/image001.jpg  →  masks/image001.jpg.png
        The mask is a grayscale image where 0 = ignore, 255 = keep.
        """
        job_id = job.job_id
        images_dir = job.images_dir
        masks_dir = job.job_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

        # Determine which classes to mask
        masking_config = self._config.masking
        if remove_classes is None:
            job_mask_config = job.config.get("masking", {})
            remove_classes_list = job_mask_config.get(
                "remove_classes",
                masking_config.remove_classes
            )
            remove_classes = set(remove_classes_list)

        # Map class names to DeepLabV3 label indices
        class_indices = set()
        for cls_name in remove_classes:
            idx = _get_deeplabv3_class_index(cls_name)
            if idx is not None:
                class_indices.add(idx)
            else:
                logger.warning(f"Unknown class '{cls_name}', skipping")

        if not class_indices:
            logger.warning("No valid classes to mask, skipping mask generation")
            return masks_dir

        # Collect image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = sorted(
            f for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        )

        if not image_files:
            logger.warning("No images found for mask generation")
            return masks_dir

        total = len(image_files)
        logger.info(f"Generating masks for {total} images, removing classes: {remove_classes}")

        await self._job_manager.update_job_progress(
            job_id,
            message=f"Generating masks (0/{total})...",
            overall_progress=3.0,
        )

        # Run inference in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._generate_masks_sync,
            job_id, image_files, masks_dir, class_indices, total, loop
        )

        logger.info(f"Mask generation complete: {masks_dir}")
        return masks_dir

    def _generate_masks_sync(
        self,
        job_id: str,
        image_files: list,
        masks_dir: Path,
        class_indices: Set[int],
        total: int,
        loop: asyncio.AbstractEventLoop,
    ):
        """Synchronous mask generation (runs in executor thread)."""
        import torch
        from PIL import Image
        import numpy as np

        self._load_model()

        for i, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = self._transforms(img).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    output = self._model(input_tensor)["out"]
                    pred = output.argmax(1).squeeze(0).cpu().numpy()

                # Build mask: 255 = keep, 0 = ignore
                mask = np.full(pred.shape, 255, dtype=np.uint8)
                for idx in class_indices:
                    mask[pred == idx] = 0

                # Save mask — COLMAP expects <image_name>.png in the mask dir
                mask_filename = img_path.name + ".png"
                mask_img = Image.fromarray(mask, mode="L")
                mask_img.save(masks_dir / mask_filename)

            except Exception as e:
                logger.error(f"Error generating mask for {img_path.name}: {e}")

            # Update progress periodically
            if (i + 1) % max(1, total // 10) == 0 or i == total - 1:
                progress = 3.0 + (i + 1) / total * 2.0  # 3% → 5%
                asyncio.run_coroutine_threadsafe(
                    self._job_manager.update_job_progress(
                        job_id,
                        message=f"Generating masks ({i + 1}/{total})...",
                        overall_progress=progress,
                    ),
                    loop,
                )

        # Free GPU memory
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
