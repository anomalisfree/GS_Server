"""Classical sky detector for outdoor frames (EXPERIMENTAL - gave only a small effect, see
CameraSet_Training_Report_2026-10-19.md).

Sky = blue-ish, bright, smooth (low gradient) and connected to the TOP edge of the image.
Conservative on purpose: sky seen through tree branches and small patches near the horizon are NOT masked,
and white house walls are not mistaken for sky.  Returns uint8: 255 = keep, 0 = sky (Brush ignores 0 in
the loss when it sits in masks/<image stem>.png next to images/).
"""
import cv2
import numpy as np


def sky_mask(bgr, dilate_px=15, work_max=480):
    h0, w0 = bgr.shape[:2]
    s = work_max / max(h0, w0)
    small = cv2.resize(bgr, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.2), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.2), cv2.CV_32F, 0, 1)
    grad = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)           # sky is nearly textureless
    blue = (H >= 92) & (H <= 135) & (S >= 35) & (V >= 110)           # clear blue sky
    haze = (S <= 70) & (V >= 190) & (grad < 6.0)                     # bright flat desaturated (clouds/haze)
    cand = (blue | haze) & (grad < 14.0)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    sky = np.zeros_like(cand)
    for i in range(1, n):
        comp = lab == i
        touches_top = stats[i, cv2.CC_STAT_TOP] <= 2
        if touches_top and stats[i, cv2.CC_STAT_AREA] > 0.004 * cand.size and (comp & blue).sum() > 0.25 * comp.sum():
            sky[comp] = 1
    sky = cv2.resize(sky, (w0, h0), interpolation=cv2.INTER_NEAREST)
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        sky = cv2.dilate(sky, k)
    return np.where(sky > 0, 0, 255).astype(np.uint8)
