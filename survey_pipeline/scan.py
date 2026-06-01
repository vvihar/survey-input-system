from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import torch
from PIL import Image

from .common import (
    DEFAULT_RESPONDENT_ID_RECT_PT,
    LOCAL_PAGES_PER_BOOKLET,
    VERSION_BASE_PAGE,
    VERSION_LABEL_SEARCH_RECT_PT,
    VERSION_LABEL_TEMPLATE_RECT_PT,
    crop_by_rect_pt,
    match_template_score,
    render_pdf_page,
)
from .digit_model import (
    extract_ink,
    load_model,
    normalize_digit_for_mnist,
    predict_digit,
    split_digit_components,
    validate_value,
)

ID_PATTERN = re.compile(r"[A-Z0-9]{2,4}-[A-Z0-9]{3,5}")


def detect_version_from_text(doc: fitz.Document, page_index: int) -> str | None:
    text = doc[page_index].get_text("text") or ""
    m = re.search(r"版\s*([ABC])", text)
    return m.group(1) if m else None


def detect_version_by_template(
    answered_pdf: Path, template_pdf: Path, answered_page_index: int, dpi: int = 150
) -> tuple[str, dict[str, float]]:
    scan_page, _ = render_pdf_page(answered_pdf, answered_page_index, dpi=dpi)
    scores: dict[str, float] = {}
    # 版表示周辺だけで照合。スキャンがずれている場合に備え、検索領域はやや広くする。
    search = crop_by_rect_pt(scan_page, VERSION_LABEL_SEARCH_RECT_PT, dpi=dpi)
    for version, base in VERSION_BASE_PAGE.items():
        tmpl_page, _ = render_pdf_page(template_pdf, base, dpi=dpi)
        tmpl = crop_by_rect_pt(tmpl_page, VERSION_LABEL_TEMPLATE_RECT_PT, dpi=dpi)
        scores[version] = match_template_score(search, tmpl)
    version = max(scores, key=scores.get)
    return version, scores


def extract_id_from_text(doc: fitz.Document, booklet_start: int) -> str | None:
    # ID欄の矩形内に印字されたテキストだけを拾う。
    # ページ全体を検索すると電話番号等を誤検出するため、範囲を限定する。
    x0, y0, x1, y1 = DEFAULT_RESPONDENT_ID_RECT_PT
    for i in range(
        booklet_start, min(booklet_start + LOCAL_PAGES_PER_BOOKLET, len(doc))
    ):
        words = doc[i].get_text("words")
        buf = []
        for w in words:
            wx0, wy0, wx1, wy1, word, *_ = w
            if wx1 >= x0 and wx0 <= x1 and wy1 >= y0 and wy0 <= y1:
                buf.append(word)
        text = "".join(buf).replace(" ", "")
        for cand in ID_PATTERN.findall(text):
            if not cand.startswith("HTTP"):
                return cand
    return None


def extract_id_by_ocr(
    answered_pdf: Path, page_index: int, dpi: int = 220
) -> str | None:
    try:
        import pytesseract
    except Exception:
        return None
    img, _ = render_pdf_page(answered_pdf, page_index, dpi=dpi)
    crop = crop_by_rect_pt(img, DEFAULT_RESPONDENT_ID_RECT_PT, dpi=dpi, pad_px=10)
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(
        pil,
        config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
    )
    m = ID_PATTERN.search(text.replace(" ", ""))
    return m.group(0) if m else None


def _inner_answer_rect(rect: list[float], margin_pt: float = 2.2) -> list[float]:
    """Return the inside of the answer box, excluding the printed frame."""
    x0, y0, x1, y1 = rect
    if x1 - x0 <= 2 * margin_pt or y1 - y0 <= 2 * margin_pt:
        return rect
    return [x0 + margin_pt, y0 + margin_pt, x1 - margin_pt, y1 - margin_pt]


def read_digit_item(
    aligned_scan_img: np.ndarray,
    template_img: np.ndarray,
    item: dict[str, Any],
    model: Any | None,
    device: torch.device,
    dpi: int,
) -> dict[str, Any]:
    # Use the inner box for recognition.  Including the frame creates strong
    # residuals after alignment and can swamp the handwritten digit.
    recog_rect = _inner_answer_rect(item["rect"], margin_pt=3.6)
    crop_scan = crop_by_rect_pt(aligned_scan_img, recog_rect, dpi=dpi, pad_px=0)
    crop_tmpl = crop_by_rect_pt(template_img, recog_rect, dpi=dpi, pad_px=0)
    binary = extract_ink(crop_scan, crop_tmpl)
    components = split_digit_components(binary)

    pred_value = ""
    confs: list[float] = []
    if model is not None and components:
        chars = []
        for comp in components:
            img28 = normalize_digit_for_mnist(comp)
            d, conf = predict_digit(model, img28, device)
            chars.append(str(d))
            confs.append(conf)
        pred_value = "".join(chars)
    ink_ratio = float(np.count_nonzero(binary) / max(1, binary.size))
    if ink_ratio < 0.0015:
        pred_value = ""
    pred_conf = float(min(confs)) if confs else None
    status = validate_value(pred_value, item)
    if pred_conf is not None and pred_conf < 0.70:
        status = "needs_review"
    return {
        "field_id": item["id"],
        "type": "digit",
        "pred_value": pred_value,
        "pred_confidence": pred_conf,
        "ink_ratio": ink_ratio,
        "status": status,
    }


def read_scanned_pdf(
    answered_pdf: Path,
    template_pdf: Path,
    layout: dict[str, Any],
    model_path: Path | None = None,
    dpi: int = 220,
    use_id_ocr: bool = True,
) -> dict[str, Any]:
    doc = fitz.open(answered_pdf)
    n_pages = len(doc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device) if model_path else None

    booklets: list[dict[str, Any]] = []
    for start in range(0, n_pages, LOCAL_PAGES_PER_BOOKLET):
        if start + LOCAL_PAGES_PER_BOOKLET > n_pages:
            break
        version = detect_version_from_text(doc, start)
        version_scores = None
        if version is None:
            version, version_scores = detect_version_by_template(
                answered_pdf, template_pdf, start, dpi=150
            )
        rid = extract_id_from_text(doc, start)
        if rid is None and use_id_ocr:
            rid = extract_id_by_ocr(answered_pdf, start, dpi=max(220, dpi))
        if rid is None:
            rid = f"booklet_{start // LOCAL_PAGES_PER_BOOKLET + 1:04d}"

        answers: dict[str, Any] = {}
        alignments: list[dict[str, Any]] = []
        for local_page in range(LOCAL_PAGES_PER_BOOKLET):
            ans_page_index = start + local_page
            tmpl_page_index = VERSION_BASE_PAGE[version] + local_page
            scan_img, _ = render_pdf_page(answered_pdf, ans_page_index, dpi=dpi)
            tmpl_img, _ = render_pdf_page(template_pdf, tmpl_page_index, dpi=dpi)
            from .common import align_ecc_affine

            aligned, score = align_ecc_affine(scan_img, tmpl_img)
            alignments.append(
                {
                    "answered_page_index": ans_page_index,
                    "template_page_index": tmpl_page_index,
                    "ecc": score,
                }
            )
            layout_page = next(
                p
                for p in layout["pages"]
                if p["template_page_index"] == tmpl_page_index
            )
            for item in layout_page.get("items", []):
                if item.get("type") != "digit":
                    continue
                answers[item["id"]] = read_digit_item(
                    aligned, tmpl_img, item, model, device, dpi
                )

        booklets.append(
            {
                "booklet_index": start // LOCAL_PAGES_PER_BOOKLET,
                "answered_page_start": start,
                "respondent_id": rid,
                "version": version,
                "version_scores": version_scores,
                "alignments": alignments,
                "answers": answers,
            }
        )
    doc.close()
    return {
        "source_pdf": str(answered_pdf),
        "template_pdf": str(template_pdf),
        "dpi": dpi,
        "booklets": booklets,
    }
