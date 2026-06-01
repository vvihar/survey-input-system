from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .common import (
    BOOKLET_META_RECT_BY_KIND,
    LOCAL_PAGES_PER_BOOKLET,
    VERSION_BASE_PAGE,
    VERSIONS,
    align_ecc_affine,
    crop_by_rect_pt,
    render_pdf_page,
    write_json,
    write_jsonl,
)
from .models import Manifest, ReviewActivityRow, ReviewItem


def _answer_lookup(
    initial_answers: dict[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for b in initial_answers.get("booklets", []):
        bi = int(b["booklet_index"])
        for fid, val in b.get("answers", {}).items():
            out[(bi, fid)] = val
    return out


def _booklet_meta(initial_answers: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(b["booklet_index"]): b for b in initial_answers.get("booklets", [])}


def _make_review_dataset_one(
    answered_pdf: Path,
    template_pdf: Path,
    layout: dict[str, Any],
    initial_answers: dict[str, Any],
    workdir: Path,
    dpi: int = 220,
) -> list[ReviewItem]:
    """Process one answered PDF and return list of review items (does NOT write to disk)."""
    crops_dir = workdir / "crops"
    contexts_dir = workdir / "contexts"
    crops_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)

    lookup = _answer_lookup(initial_answers)
    meta = _booklet_meta(initial_answers)
    rows: list[ReviewItem] = []
    meta_dir = workdir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    source_pdf = str(answered_pdf)
    n_booklets = len(initial_answers.get("booklets", []))
    for booklet_index in range(n_booklets):
        bmeta = meta.get(booklet_index)
        if bmeta is None:
            continue
        version = bmeta["version"]
        respondent_id = bmeta["respondent_id"]
        page_start = int(bmeta["answered_page_start"])
        bkl_source_pdf = bmeta.get("source_pdf", source_pdf)

        meta_scan_img, _ = render_pdf_page(answered_pdf, page_start, dpi=dpi)
        rid_crop = crop_by_rect_pt(
            meta_scan_img,
            BOOKLET_META_RECT_BY_KIND["respondent_id"],
            dpi=dpi,
            pad_px=12,
        )
        ver_crop = crop_by_rect_pt(
            meta_scan_img, BOOKLET_META_RECT_BY_KIND["version"], dpi=dpi, pad_px=12
        )
        rid_meta_path = meta_dir / f"b{booklet_index:04d}_respondent_id.png"
        ver_meta_path = meta_dir / f"b{booklet_index:04d}_version.png"
        cv2.imwrite(str(rid_meta_path), rid_crop)
        cv2.imwrite(str(ver_meta_path), ver_crop)

        for local_page in range(LOCAL_PAGES_PER_BOOKLET):
            answered_page = page_start + local_page
            template_page = VERSION_BASE_PAGE[version] + local_page
            scan_img, _ = render_pdf_page(answered_pdf, answered_page, dpi=dpi)
            tmpl_img, _ = render_pdf_page(template_pdf, template_page, dpi=dpi)
            aligned, ecc = align_ecc_affine(scan_img, tmpl_img)
            layout_page = next(
                p for p in layout["pages"] if p["template_page_index"] == template_page
            )

            for item in layout_page.get("items", []):
                field_id = item["id"]
                stem = f"b{booklet_index:04d}_{respondent_id}_{field_id}"
                crop_path = crops_dir / f"{stem}.png"
                context_path = contexts_dir / f"{stem}.png"
                crop = crop_by_rect_pt(
                    aligned,
                    item.get("rect", item.get("context_rect")),
                    dpi=dpi,
                    pad_px=12,
                )
                context = crop_by_rect_pt(
                    aligned,
                    item.get("context_rect", item.get("rect")),
                    dpi=dpi,
                    pad_px=12,
                )
                cv2.imwrite(str(crop_path), crop)
                cv2.imwrite(str(context_path), context)
                pred = lookup.get((booklet_index, field_id), {})
                row = ReviewItem(
                    item_uid=stem,
                    booklet_index=booklet_index,
                    respondent_id=respondent_id,
                    version=version,
                    source_pdf=bkl_source_pdf,
                    answered_page_index=answered_page,
                    template_page_index=template_page,
                    local_page_index=local_page,
                    field_id=field_id,
                    type=item["type"],
                    label=item.get("label", field_id),
                    min=item.get("min"),
                    max=item.get("max"),
                    multiple=bool(item.get("multiple", False)),
                    day=item.get("day"),
                    rows=item.get("rows"),
                    columns=list(item.get("columns", []))
                    if item.get("columns") is not None
                    else None,
                    scenario_macro=item.get("scenario_macro"),
                    scenario_text=item.get("scenario_text"),
                    crop_path=str(crop_path.relative_to(workdir)),
                    context_path=str(context_path.relative_to(workdir)),
                    pred_value=pred.get("pred_value"),
                    pred_confidence=pred.get("pred_confidence"),
                    value=pred.get("pred_value") if item["type"] == "digit" else None,
                    status=pred.get(
                        "status", "needs_review" if item["type"] != "digit" else "blank"
                    ),
                    ecc=ecc,
                    meta_respondent_id_path=str(rid_meta_path.relative_to(workdir)),
                    meta_version_path=str(ver_meta_path.relative_to(workdir)),
                    activity_rows=[
                        ReviewActivityRow(row_no=i + 1)
                        for i in range(int(item.get("rows", 7)))
                    ]
                    if item["type"] == "activity_day"
                    else [],
                )
                rows.append(row)

    return rows


def make_review_dataset(
    answered_pdf: Path,
    template_pdf: Path,
    layout: dict[str, Any],
    initial_answers: dict[str, Any],
    workdir: Path,
    dpi: int = 220,
) -> dict[str, Any]:
    """Process a single answered PDF. For multiple PDFs, use make_review_datasets()."""
    rows = _make_review_dataset_one(
        answered_pdf, template_pdf, layout, initial_answers, workdir, dpi
    )

    manifest = Manifest(
        answered_pdf=str(answered_pdf),
        template_pdf=str(template_pdf),
        dpi=dpi,
        n_items=len(rows),
        review_items="review_items.jsonl",
    )
    write_json(workdir / "manifest.json", manifest.model_dump(mode="json"))
    write_json(workdir / "initial_answers.json", initial_answers)
    write_json(workdir / "layout.json", layout)
    write_jsonl(
        workdir / "review_items.jsonl", [x.model_dump(mode="json") for x in rows]
    )
    return manifest.model_dump(mode="json")


def make_review_datasets(
    answered_pdfs: list[Path],
    template_pdf: Path,
    layout: dict[str, Any],
    merged_initial_answers: dict[str, Any],
    workdir: Path,
    dpi: int = 220,
) -> dict[str, Any]:
    """Process multiple answered PDFs into a single review workdir.

    ``merged_initial_answers`` is the combined output of ``read_scanned_pdfs()``
    where each booklet dict has a ``source_pdf`` key.
    """
    all_rows: list[ReviewItem] = []

    # Group booklets by source_pdf
    booklets_by_pdf: dict[str, list[dict[str, Any]]] = {}
    for b in merged_initial_answers.get("booklets", []):
        sp = b.get("source_pdf", "")
        booklets_by_pdf.setdefault(sp, []).append(b)

    for pdf in answered_pdfs:
        sp = str(pdf)
        pdf_booklets = booklets_by_pdf.get(sp, [])
        if not pdf_booklets:
            continue
        # Build a partial initial_answers dict for this PDF
        partial_answers: dict[str, Any] = {
            "source_pdf": sp,
            "template_pdf": merged_initial_answers.get(
                "template_pdf", str(template_pdf)
            ),
            "dpi": merged_initial_answers.get("dpi", dpi),
            "booklets": pdf_booklets,
        }
        rows = _make_review_dataset_one(
            pdf, template_pdf, layout, partial_answers, workdir, dpi
        )
        all_rows.extend(rows)

    source_pdfs = [str(p) for p in answered_pdfs]

    manifest = Manifest(
        answered_pdf=source_pdfs if len(source_pdfs) > 1 else source_pdfs[0],
        template_pdf=str(template_pdf),
        dpi=dpi,
        n_items=len(all_rows),
        review_items="review_items.jsonl",
    )
    write_json(workdir / "manifest.json", manifest.model_dump(mode="json"))
    write_json(workdir / "initial_answers.json", merged_initial_answers)
    write_json(workdir / "layout.json", layout)
    write_jsonl(
        workdir / "review_items.jsonl", [x.model_dump(mode="json") for x in all_rows]
    )
    return manifest.model_dump(mode="json")


def update_booklet_meta(
    initial_answers: dict[str, Any],
    booklet_index: int,
    respondent_id: str,
    version: str,
) -> dict[str, Any]:
    updated = dict(initial_answers)
    booklets = [dict(b) for b in initial_answers.get("booklets", [])]
    for b in booklets:
        if int(b["booklet_index"]) == booklet_index:
            if respondent_id:
                b["respondent_id"] = respondent_id
            if version in VERSIONS:
                b["version"] = version
            break
    updated["booklets"] = booklets
    return updated
