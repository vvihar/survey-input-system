from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


def load_model(model_path: Path | None, device: torch.device) -> nn.Module | None:
    """Load a digit model checkpoint.

    PyTorch 2.6+ defaults torch.load(..., weights_only=True).  Some older
    checkpoints saved by train_digit_model.py contained pathlib.Path objects
    in the metadata, which are rejected by the restricted weights-only loader.
    For local checkpoints that you created yourself, falling back to
    weights_only=False is acceptable.  Do not use untrusted .pt files.
    """
    if model_path is None:
        return None

    model = MnistNet().to(device)

    try:
        obj = torch.load(model_path, map_location=device)
    except Exception as exc:
        msg = str(exc)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        # Only safe for checkpoints you created locally / trust.
        obj = torch.load(model_path, map_location=device, weights_only=False)

    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _match_background_level(
    scan_gray: npt.NDArray[np.uint8], tmpl_gray: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Match paper-background brightness of scan_crop to template_crop.

    Scans are often globally darker than the rendered template.  If we simply
    compute template - scan, the whole crop may become foreground.  Estimate the
    paper background from bright pixels and shift the scan before differencing.
    """
    scan_f: npt.NDArray[np.float32] = scan_gray.astype(np.float32)
    tmpl_f: npt.NDArray[np.float32] = tmpl_gray.astype(np.float32)

    scan_bg = scan_f[scan_f >= np.percentile(scan_f, 65)]
    tmpl_bg = tmpl_f[tmpl_f >= np.percentile(tmpl_f, 65)]
    if scan_bg.size and tmpl_bg.size:
        delta = float(np.median(tmpl_bg) - np.median(scan_bg))
        scan_f = np.clip(scan_f + delta, 0, 255)
    return scan_f.astype(np.uint8)


def _remove_small_components(
    binary: npt.NDArray[np.uint8], min_area: int = 14
) -> npt.NDArray[np.uint8]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
    return out


def extract_ink(
    scan_crop: npt.NDArray[np.uint8], tmpl_crop: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Extract handwritten ink from an answer box.

    This version is intentionally more aggressive about removing the printed
    answer-box frame and the pale "解答欄" placeholder.  The previous method used
    absdiff(scan, template), which also preserves small alignment residuals from
    printed elements.  Those residuals can dominate the tiny digit crop and make
    MNIST accuracy collapse.

    Returned image: uint8 binary, black background and white handwritten ink.
    """
    if scan_crop.shape[:2] != tmpl_crop.shape[:2]:
        tmpl_crop = cv2.resize(tmpl_crop, (scan_crop.shape[1], scan_crop.shape[0]))  # type: ignore[assignment]

    scan_gray: npt.NDArray[np.uint8] = cv2.cvtColor(scan_crop, cv2.COLOR_BGR2GRAY)  # type: ignore[assignment]
    tmpl_gray: npt.NDArray[np.uint8] = cv2.cvtColor(tmpl_crop, cv2.COLOR_BGR2GRAY)  # type: ignore[assignment]

    scan_gray = _match_background_level(scan_gray, tmpl_gray)
    scan_blur = cv2.GaussianBlur(scan_gray, (3, 3), 0)
    tmpl_blur = cv2.GaussianBlur(tmpl_gray, (3, 3), 0)

    # Handwriting makes the scanned image darker than the blank template.
    # Use signed darkening, not absdiff, so bright-side edge residuals from the
    # printed frame are not treated as ink.
    darkening = cv2.subtract(tmpl_blur, scan_blur)

    # Conservative fixed threshold.  Otsu is unstable on small mostly-white crops.
    _, binary = cv2.threshold(darkening, 18, 255, cv2.THRESH_BINARY)

    # Remove pixels that already belong to printed template ink: frame lines,
    # QR fragments if accidentally included, and the pale "解答欄" placeholder.
    template_ink = (tmpl_gray < 248).astype(np.uint8) * 255
    template_ink = cv2.dilate(template_ink, np.ones((3, 3), np.uint8), iterations=2)  # type: ignore[assignment]
    binary[template_ink > 0] = 0

    # Reconnect handwriting strokes that were broken by placeholder removal.
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1
    )  # type: ignore[assignment]
    binary = cast(npt.NDArray[np.uint8], binary)
    binary = _remove_small_components(binary, min_area=14)
    binary = _strip_dirty_border(binary)
    binary = _suppress_edge_artifacts(binary)
    binary = _remove_small_components(binary, min_area=14)
    return binary


def _suppress_edge_artifacts(binary: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Remove residual vertical frame fragments near crop edges.

    The answer box frame can survive template subtraction as a thin vertical
    component.  MNIST then often reads this component as an extra "1", producing
    values such as "11" or "13".  We remove components that are simultaneously:
      - close to the left/right edge,
      - thin and tall,
      - roughly vertical.

    A genuine handwritten "1" in the middle of the box is retained.
    """
    H, W = binary.shape[:2]
    if H == 0 or W == 0:
        return binary

    out = np.zeros_like(binary)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    edge_margin = max(3, int(round(W * 0.12)))
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        near_left_or_right = x <= edge_margin or (x + w) >= (W - edge_margin)
        thin = w <= max(3, int(round(W * 0.12)))
        tall = h >= int(round(H * 0.42))
        very_vertical = h >= max(8, 2 * w)

        # Edge-localized vertical stroke: likely residual from the box frame.
        if near_left_or_right and thin and tall and very_vertical:
            continue

        # Fragments touching the very top/bottom near an edge are also usually
        # frame remnants after affine alignment.
        touches_horizontal_edge = y <= 1 or (y + h) >= H - 1
        if near_left_or_right and touches_horizontal_edge and area >= 8:
            continue

        out[labels == i] = 255

    return out


def _strip_dirty_border(binary: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Clear only border columns/rows that are dominated by residual ink."""
    H, W = binary.shape[:2]
    if H == 0 or W == 0:
        return binary

    out = binary.copy()
    bw = max(2, int(round(W * 0.06)))
    bh = max(2, int(round(H * 0.06)))

    # If many pixels exist in a border band, it is usually the printed frame.
    # Clearing the band is safer than letting it become an extra "1".
    if np.count_nonzero(out[:, :bw]) > 0.20 * H * bw:
        out[:, :bw] = 0
    if np.count_nonzero(out[:, W - bw :]) > 0.20 * H * bw:
        out[:, W - bw :] = 0
    if np.count_nonzero(out[:bh, :]) > 0.20 * W * bh:
        out[:bh, :] = 0
    if np.count_nonzero(out[H - bh :, :]) > 0.20 * W * bh:
        out[H - bh :, :] = 0

    return out


def split_digit_components(
    binary: npt.NDArray[np.uint8],
) -> list[npt.NDArray[np.uint8]]:
    binary = _strip_dirty_border(binary)
    binary = _suppress_edge_artifacts(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 12 or h < H * 0.18 or w < 2:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        x, y, w, h = box
        px, py, pw, ph = merged[-1]
        gap = x - (px + pw)
        overlap = min(py + ph, y + h) - max(py, y)
        if gap <= 3 and overlap > 0:
            nx0, ny0 = min(px, x), min(py, y)
            nx1, ny1 = max(px + pw, x + w), max(py + ph, y + h)
            merged[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0)
        else:
            merged.append(box)
    crops: list[np.ndarray] = []
    for x, y, w, h in merged:
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crops.append(binary[y0:y1, x0:x1])
    return crops


def normalize_digit_for_mnist(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    img = img[y : y + h, x : x + w]

    # Keep MNIST convention: digit fits into roughly 20x20 and is centered in 28x28.
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas: npt.NDArray[np.uint8] = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    # MNIST digits are approximately center-of-mass normalized.  This helps when
    # answer-box crops are slightly off-center after scanning/alignment.
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))
        M = cast(
            npt.NDArray[np.float32],
            np.float32([[1, 0, shift_x], [0, 1, shift_y]]),  # type: ignore[arg-type]
        )
        canvas = cv2.warpAffine(canvas, M, (28, 28), borderValue=0)  # type: ignore[assignment]

    return canvas.astype(np.float32) / 255.0


def predict_digit(
    model: nn.Module, img28: npt.NDArray[np.float32], device: torch.device
) -> tuple[int, float]:
    x = torch.from_numpy(img28).float().unsqueeze(0).unsqueeze(0)
    x = (x - 0.1307) / 0.3081
    x = x.to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        conf, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf.item())


def validate_value(value: str | None, item: dict) -> str:
    if value is None or value == "":
        return "blank"
    if not value.isdigit():
        return "needs_review"
    lo, hi = item.get("min"), item.get("max")
    if item.get("multiple"):
        if lo is None or hi is None:
            return "auto"
        return (
            "auto"
            if all(int(ch) >= lo and int(ch) <= hi for ch in value)
            else "needs_review"
        )
    iv = int(value)
    if lo is not None and iv < lo:
        return "needs_review"
    if hi is not None and iv > hi:
        return "needs_review"
    return "auto"
