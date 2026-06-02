from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import fitz
import numpy as np
import numpy.typing as npt

from .models import InitialAnswers, LayoutDocument, ReviewItem, Version

VERSION_BASE_PAGE: dict[Version, int] = {"A": 0, "B": 4, "C": 8}
VERSIONS = ("A", "B", "C")
LOCAL_PAGES_PER_BOOKLET = 4

# PyMuPDF PDF point coordinates. Origin: top-left.
VERSION_LABEL_TEMPLATE_RECT_PT = [0.0, 785.0, 100.0, 841.89]
VERSION_LABEL_SEARCH_RECT_PT = [0.0, 745.0, 100.0, 841.89]
DEFAULT_RESPONDENT_ID_RECT_PT = [35.0, 615.0, 330.0, 740.0]

BOOKLET_META_RECT_BY_KIND = {
    "respondent_id": DEFAULT_RESPONDENT_ID_RECT_PT,
    "version": VERSION_LABEL_SEARCH_RECT_PT,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_layout(path: Path) -> LayoutDocument:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "schema" in data and "schema_data" not in data:
        data["schema_data"] = data.pop("schema")
    return LayoutDocument.model_validate(data)


def read_initial_answers(path: Path) -> InitialAnswers:
    return InitialAnswers.model_validate_json(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_review_items(path: Path) -> Iterable[ReviewItem]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield ReviewItem.model_validate_json(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rect_to_list(rect: fitz.Rect) -> list[float]:
    return [round(rect.x0, 3), round(rect.y0, 3), round(rect.x1, 3), round(rect.y1, 3)]


def inflate_rect(
    rect: list[float], dx: float, dy: float, page_w: float, page_h: float
) -> list[float]:
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


def render_pdf_page(
    pdf_path: Path, page_index: int, dpi: int
) -> tuple[npt.NDArray[np.uint8], fitz.Rect]:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img: npt.NDArray[np.uint8] = (
        np.frombuffer(pix.samples, dtype=np.uint8)
        .reshape(pix.height, pix.width, 3)
        .copy()
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    rect = fitz.Rect(page.rect)
    doc.close()
    return img, rect


def resize_uint8(
    img: npt.NDArray[np.uint8], size: tuple[int, int]
) -> npt.NDArray[np.uint8]:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.uint8)


def warp_affine_uint8(
    img: npt.NDArray[np.uint8], warp: npt.NDArray[np.float32], size: tuple[int, int]
) -> npt.NDArray[np.uint8]:
    return cv2.warpAffine(
        img,
        warp,
        size,
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    ).astype(np.uint8)


def rect_pt_to_px(
    rect_pt: list[float], dpi: int, pad_px: int = 0
) -> tuple[int, int, int, int]:
    scale = dpi / 72.0
    x0, y0, x1, y1 = rect_pt
    return (
        int(round(x0 * scale)) - pad_px,
        int(round(y0 * scale)) - pad_px,
        int(round(x1 * scale)) + pad_px,
        int(round(y1 * scale)) + pad_px,
    )


def crop_px(
    img: npt.NDArray[np.uint8], rect_px: tuple[int, int, int, int]
) -> npt.NDArray[np.uint8]:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect_px
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    return img[y0:y1, x0:x1]


def crop_by_rect_pt(
    img: npt.NDArray[np.uint8], rect_pt: list[float], dpi: int, pad_px: int = 0
) -> npt.NDArray[np.uint8]:
    return crop_px(img, rect_pt_to_px(rect_pt, dpi, pad_px))


def to_gray_float(img_bgr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.float32) / 255.0


def align_ecc_affine(
    scan_img: npt.NDArray[np.uint8], template_img: npt.NDArray[np.uint8]
) -> tuple[npt.NDArray[np.uint8], float]:
    aligned, score, _ = align_ecc_affine_with_matrix(scan_img, template_img)
    return aligned, score


def align_ecc_affine_with_matrix(
    scan_img: npt.NDArray[np.uint8], template_img: npt.NDArray[np.uint8]
) -> tuple[npt.NDArray[np.uint8], float, list[list[float]] | None]:
    h, w = template_img.shape[:2]
    if scan_img.shape[:2] != (h, w):
        scan_img = resize_uint8(scan_img, (w, h))

    scan_gray: npt.NDArray[np.float32] = to_gray_float(scan_img).astype(np.float32)
    tmpl_gray: npt.NDArray[np.float32] = to_gray_float(template_img).astype(np.float32)
    warp: npt.NDArray[np.float32] = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 800, 1e-6)

    try:
        cc, warp = cv2.findTransformECC(  # type: ignore[assignment]
            tmpl_gray,
            scan_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=None,
        )
        aligned: npt.NDArray[np.uint8] = warp_affine_uint8(scan_img, warp, (w, h))
        return aligned, float(cc), warp.astype(float).tolist()
    except cv2.error:
        fallback: npt.NDArray[np.uint8] = resize_uint8(scan_img, (w, h))
        return fallback, -1.0, None


def match_template_score(
    search_img: npt.NDArray[np.uint8], template_img: npt.NDArray[np.uint8]
) -> float:
    search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    if (
        search_gray.shape[0] < tmpl_gray.shape[0]
        or search_gray.shape[1] < tmpl_gray.shape[1]
    ):
        return -1.0
    res = cv2.matchTemplate(search_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)
