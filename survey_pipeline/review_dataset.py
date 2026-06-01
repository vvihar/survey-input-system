from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .common import (
    LOCAL_PAGES_PER_BOOKLET,
    VERSION_BASE_PAGE,
    align_ecc_affine,
    crop_by_rect_pt,
    read_json,
    render_pdf_page,
    write_json,
    write_jsonl,
)


def _answer_lookup(initial_answers: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for b in initial_answers.get("booklets", []):
        bi = int(b["booklet_index"])
        for fid, val in b.get("answers", {}).items():
            out[(bi, fid)] = val
    return out


def _booklet_meta(initial_answers: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(b["booklet_index"]): b for b in initial_answers.get("booklets", [])}


def make_review_dataset(
    answered_pdf: Path,
    template_pdf: Path,
    layout: dict[str, Any],
    initial_answers: dict[str, Any],
    workdir: Path,
    dpi: int = 220,
) -> dict[str, Any]:
    crops_dir = workdir / "crops"
    contexts_dir = workdir / "contexts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)

    lookup = _answer_lookup(initial_answers)
    meta = _booklet_meta(initial_answers)
    rows: list[dict[str, Any]] = []

    n_booklets = len(initial_answers.get("booklets", []))
    for booklet_index in range(n_booklets):
        bmeta = meta[booklet_index]
        version = bmeta["version"]
        respondent_id = bmeta["respondent_id"]
        page_start = int(bmeta["answered_page_start"])

        for local_page in range(LOCAL_PAGES_PER_BOOKLET):
            answered_page = page_start + local_page
            template_page = VERSION_BASE_PAGE[version] + local_page
            scan_img, _ = render_pdf_page(answered_pdf, answered_page, dpi=dpi)
            tmpl_img, _ = render_pdf_page(template_pdf, template_page, dpi=dpi)
            aligned, ecc = align_ecc_affine(scan_img, tmpl_img)
            layout_page = next(p for p in layout["pages"] if p["template_page_index"] == template_page)

            for item in layout_page.get("items", []):
                field_id = item["id"]
                stem = f"b{booklet_index:04d}_{respondent_id}_{field_id}"
                crop_path = crops_dir / f"{stem}.png"
                context_path = contexts_dir / f"{stem}.png"
                crop = crop_by_rect_pt(aligned, item.get("rect", item.get("context_rect")), dpi=dpi, pad_px=12)
                context = crop_by_rect_pt(aligned, item.get("context_rect", item.get("rect")), dpi=dpi, pad_px=12)
                cv2.imwrite(str(crop_path), crop)
                cv2.imwrite(str(context_path), context)
                pred = lookup.get((booklet_index, field_id), {})
                row = {
                    "item_uid": stem,
                    "booklet_index": booklet_index,
                    "respondent_id": respondent_id,
                    "version": version,
                    "answered_page_index": answered_page,
                    "template_page_index": template_page,
                    "local_page_index": local_page,
                    "field_id": field_id,
                    "type": item["type"],
                    "label": item.get("label", field_id),
                    "min": item.get("min"),
                    "max": item.get("max"),
                    "multiple": item.get("multiple", False),
                    "day": item.get("day"),
                    "rows": item.get("rows"),
                    "columns": item.get("columns"),
                    "scenario_macro": item.get("scenario_macro"),
                    "scenario_text": item.get("scenario_text"),
                    "crop_path": str(crop_path.relative_to(workdir)),
                    "context_path": str(context_path.relative_to(workdir)),
                    "pred_value": pred.get("pred_value"),
                    "pred_confidence": pred.get("pred_confidence"),
                    "value": pred.get("pred_value") if item["type"] == "digit" else None,
                    "status": pred.get("status", "needs_review" if item["type"] != "digit" else "blank"),
                    "ecc": ecc,
                }
                if item["type"] == "activity_day":
                    row["activity_rows"] = [
                        {"row_no": i + 1, "dep_time": "", "dep_place": "", "arr_time": "", "arr_place": "", "purpose": "", "mode": "", "tolerance_min": ""}
                        for i in range(int(item.get("rows", 7)))
                    ]
                rows.append(row)

    manifest = {
        "answered_pdf": str(answered_pdf),
        "template_pdf": str(template_pdf),
        "dpi": dpi,
        "n_items": len(rows),
        "review_items": "review_items.jsonl",
    }
    write_json(workdir / "manifest.json", manifest)
    write_json(workdir / "initial_answers.json", initial_answers)
    write_json(workdir / "layout.json", layout)
    write_jsonl(workdir / "review_items.jsonl", rows)
    return manifest
