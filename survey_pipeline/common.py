from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import fitz
import numpy as np

VERSION_BASE_PAGE = {"A": 0, "B": 4, "C": 8}
VERSIONS = ("A", "B", "C")
LOCAL_PAGES_PER_BOOKLET = 4

# PyMuPDF PDF point coordinates. Origin: top-left.
VERSION_LABEL_TEMPLATE_RECT_PT = [0.0, 785.0, 100.0, 841.89]
VERSION_LABEL_SEARCH_RECT_PT = [0.0, 745.0, 190.0, 841.89]
DEFAULT_RESPONDENT_ID_RECT_PT = [35.0, 615.0, 330.0, 720.0]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rect_to_list(rect: fitz.Rect) -> list[float]:
    return [round(rect.x0, 3), round(rect.y0, 3), round(rect.x1, 3), round(rect.y1, 3)]


def inflate_rect(rect: list[float], dx: float, dy: float, page_w: float, page_h: float) -> list[float]:
    x0, y0, x1, y1 = rect
    return [
        max(0.0, x0 - dx),
        max(0.0, y0 - dy),
        min(page_w, x1 + dx),
        min(page_h, y1 + dy),
    ]


def union_rects(rects: list[list[float]]) -> list[float]:
    return [
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    ]


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int) -> tuple[np.ndarray, fitz.Rect]:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rect = fitz.Rect(page.rect)
    doc.close()
    return img, rect


def rect_pt_to_px(rect_pt: list[float], dpi: int, pad_px: int = 0) -> tuple[int, int, int, int]:
    scale = dpi / 72.0
    x0, y0, x1, y1 = rect_pt
    return (
        int(round(x0 * scale)) - pad_px,
        int(round(y0 * scale)) - pad_px,
        int(round(x1 * scale)) + pad_px,
        int(round(y1 * scale)) + pad_px,
    )


def crop_px(img: np.ndarray, rect_px: tuple[int, int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect_px
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    return img[y0:y1, x0:x1]


def crop_by_rect_pt(img: np.ndarray, rect_pt: list[float], dpi: int, pad_px: int = 0) -> np.ndarray:
    return crop_px(img, rect_pt_to_px(rect_pt, dpi, pad_px))


def to_gray_float(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.float32) / 255.0


def align_ecc_affine(scan_img: np.ndarray, template_img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = template_img.shape[:2]
    if scan_img.shape[:2] != (h, w):
        scan_img = cv2.resize(scan_img, (w, h), interpolation=cv2.INTER_AREA)

    scan_gray = to_gray_float(scan_img)
    tmpl_gray = to_gray_float(template_img)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 800, 1e-6)

    try:
        cc, warp = cv2.findTransformECC(
            tmpl_gray,
            scan_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        aligned = cv2.warpAffine(
            scan_img,
            warp,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return aligned, float(cc)
    except cv2.error:
        return cv2.resize(scan_img, (w, h), interpolation=cv2.INTER_AREA), -1.0


def match_template_score(search_img: np.ndarray, template_img: np.ndarray) -> float:
    search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    if search_gray.shape[0] < tmpl_gray.shape[0] or search_gray.shape[1] < tmpl_gray.shape[1]:
        return -1.0
    res = cv2.matchTemplate(search_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)
