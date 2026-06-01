from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import fitz

from .common import VERSION_BASE_PAGE, inflate_rect, rect_to_list, union_rects
from .models import (
    VERSION_ADAPTER,
    Layout,
    LayoutItem,
    LayoutPage,
    TexSchema,
)
from .tex_schema import build_tex_schema


def _is_digit_answer_box(rect: fitz.Rect, drawing: dict[str, Any]) -> bool:
    # \digitboxes の1マス。約30pt角。rating5の約16.6pt角とは分離できる。
    return (
        27.0 <= rect.width <= 33.0
        and 27.0 <= rect.height <= 33.0
        and drawing.get("fill") is None
    )


def _digit_boxes(page: fitz.Page) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is not None and _is_digit_answer_box(r, d):
            rects.append(r)
    # A3横4段組の読順: 左から右、各列で上から下。
    return sorted(rects, key=lambda r: (round(r.x0, 1), round(r.y0, 1)))


def _is_rating_cell(rect: fitz.Rect) -> bool:
    return 14.0 <= rect.width <= 19.0 and 14.0 <= rect.height <= 19.0


def _rating_rows(page: fitz.Page) -> list[dict[str, Any]]:
    cells: list[fitz.Rect] = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is not None and _is_rating_cell(r):
            cells.append(r)

    rows: list[list[fitz.Rect]] = []
    for r in sorted(cells, key=lambda r: (r.y0, r.x0)):
        if not rows or abs(rows[-1][0].y0 - r.y0) > 2.0:
            rows.append([r])
        else:
            rows[-1].append(r)

    # A row grouping by y can contain more than one scale if columns happen to align
    # horizontally. Split each y-band into consecutive 5-cell scales.
    scale_rows: list[list[fitz.Rect]] = []
    for row in rows:
        row = sorted(row, key=lambda r: r.x0)
        current: list[fitz.Rect] = []
        for cell in row:
            if current and cell.x0 - current[-1].x1 > 12.0:
                if len(current) == 5:
                    scale_rows.append(current)
                current = []
            current.append(cell)
            if len(current) == 5:
                scale_rows.append(current)
                current = []
        if len(current) == 5:
            scale_rows.append(current)

    candidates: list[dict[str, Any]] = []
    words = cast(list[list[Any]], page.get_text("words"))
    for row in scale_rows:
        row = sorted(row, key=lambda r: r.x0)
        cells_pt = [rect_to_list(r) for r in row]
        row_rect = union_rects(cells_pt)
        # 左側の条件文も見えるようにする。設問全体のcropではここを使う。
        context_rect = inflate_rect(
            row_rect, dx=300.0, dy=85.0, page_w=page.rect.width, page_h=page.rect.height
        )
        x0, y0, x1, y1 = context_rect
        context_words = []
        for w in words:
            wx0, wy0, wx1, wy1, text, *_ = w
            if wx1 >= x0 and wx0 <= x1 and wy1 >= y0 and wy0 <= y1:
                context_words.append(text)
        context_text = "".join(context_words)

        # 「記入例」の判定は広い context_text で行わない。
        #
        # 広い context_text には「記入例」だけでなく、カード本文中の
        # 「例：自宅周辺」なども入りうる。そのため、広い範囲に「例」が
        # 含まれるだけで本設問の5段階欄を誤って example 扱いしてしまう。
        #
        # 実際の記入例では、5段階セルの「1」の直上に小さく「例」が
        # 印字されている。そこで、1番セルの直上近傍だけを見て判定する。
        first_cell = row[0]
        tight_example = False
        for w in words:
            wx0, wy0, wx1, wy1, text, *_ = w
            if text != "例":
                continue
            if (
                abs(wx0 - first_cell.x0) <= 18.0
                and first_cell.y0 - 18.0 <= wy0 <= first_cell.y0 + 2.0
            ):
                tight_example = True
                break

        candidates.append(
            {
                "cells": cells_pt,
                "rect": row_rect,
                "context_rect": context_rect,
                "context_text": context_text[:300],
                "example_like": tight_example,
            }
        )

    # 読順用。scale自体はカード右端にあるため、xで列、yで列内順に並べる。
    return sorted(
        candidates, key=lambda it: (round(it["rect"][0] / 250), it["rect"][1])
    )


def _find_activity_blocks(page: fitz.Page, days: list[str]) -> list[dict[str, Any]]:
    # DayTableはtcolorboxなので大きな矩形として検出できる。
    large_rects: list[fitz.Rect] = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r and r.width > 250 and r.height > 160:
            large_rects.append(r)

    result: list[dict[str, Any]] = []
    for day in days:
        hits = page.search_for(day)
        for hit in hits:
            # day titleを含み、かつ最も小さい大矩形を選ぶ。
            containing = [r for r in large_rects if r.contains(hit)]
            if not containing:
                continue
            box = min(containing, key=lambda r: r.width * r.height)
            outer = rect_to_list(box)
            # タイトルを含む外枠と、入力表部分を分ける。表部分は内側矩形を優先。
            inner_candidates = [
                r
                for r in large_rects
                if box.contains(r)
                and r != box
                and r.height < box.height
                and r.width > box.width * 0.85
            ]
            inner = (
                rect_to_list(max(inner_candidates, key=lambda r: r.height))
                if inner_candidates
                else inflate_rect(outer, -4, -22, page.rect.width, page.rect.height)
            )
            result.append(
                {
                    "day": day,
                    "rect": outer,
                    "table_rect": inner,
                    "rows": 7,
                    "columns": [
                        "dep_time",
                        "dep_place",
                        "arr_time",
                        "arr_place",
                        "purpose",
                        "mode",
                        "tolerance_min",
                    ],
                }
            )
    # 重複除去
    seen = set()
    uniq: list[dict[str, Any]] = []
    for x in result:
        key = (x["day"], tuple(round(v, 1) for v in x["rect"]))
        if key not in seen:
            uniq.append(x)
            seen.add(key)
    return sorted(uniq, key=lambda it: (it["rect"][1], it["rect"][0]))


def build_layout(template_pdf: Path, tex_path: Path) -> Layout:
    schema: TexSchema = build_tex_schema(tex_path)
    digit_questions = schema.digit_questions
    scenario_sets = schema.scenario_sets
    activity_days = [x.day for x in schema.activities]

    doc: fitz.Document = fitz.open(template_pdf)
    pages: list[LayoutPage] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)

        version = next(
            v for v, base in VERSION_BASE_PAGE.items() if base <= page_index <= base + 3
        )
        local_page_index = (
            page_index - VERSION_BASE_PAGE[VERSION_ADAPTER.validate_python(version)]
        )
        page_w, page_h = float(page.rect.width), float(page.rect.height)
        items: list[LayoutItem] = []

        # q1_1〜q3_10: answerbox由来の数字欄。
        boxes = _digit_boxes(page)
        # このTeXでは数字欄はlocal page 1と2のみにある。検出順にTeX上の質問順を対応させる。
        # page 1: q1_1〜q3_4、page 2: q3_5〜q3_10。
        if local_page_index == 0:
            qs = digit_questions[:16]
        elif local_page_index == 1:
            qs = digit_questions[16:]
        else:
            qs = []
        for q, box in zip(qs, boxes):
            rect = rect_to_list(box)
            items.append(
                LayoutItem.model_validate(
                    {
                        "id": q.id,
                        "type": q.type,
                        "label": q.label,
                        "version": version,
                        "template_page_index": page_index,
                        "local_page_index": local_page_index,
                        "rect": rect,
                        "context_rect": inflate_rect(
                            rect, 180.0, 110.0, page_w, page_h
                        ),
                        "min": q.min,
                        "max": q.max,
                        "multiple": q.multiple,
                        "choices": q.choices,
                    }
                )
            )

        # 5段階評価欄。TeX上のscenario setをメタデータとして付ける。
        if local_page_index == 1:
            rows = _rating_rows(page)
            non_examples = [r for r in rows if not r.get("example_like")]
            # ヒューリスティックで例が落ちすぎる場合は、先頭2件を例扱いにする。
            if len(non_examples) < 12 and len(rows) >= 14:
                non_examples = rows[2:14]
            scenario_items = scenario_sets.get(version, [])
            for i, row in enumerate(non_examples[: len(scenario_items)]):
                meta = scenario_items[i]
                items.append(
                    LayoutItem.model_validate(
                        {
                            "id": meta.id,
                            "type": meta.type,
                            "part": meta.part,
                            "label": meta.label,
                            "scenario_macro": meta.scenario_macro,
                            "scenario_text": meta.scenario_text,
                            "min": meta.min,
                            "max": meta.max,
                            "version": version,
                            "template_page_index": page_index,
                            "local_page_index": local_page_index,
                            "rect": row["rect"],
                            "context_rect": row["context_rect"],
                            "cells": row["cells"],
                            "context_text": row.get("context_text", ""),
                        }
                    )
                )

        # 活動表。日単位で切り出し、GUIで7行入力。
        if local_page_index in (2, 3):
            for block in _find_activity_blocks(page, activity_days):
                items.append(
                    LayoutItem.model_validate(
                        {
                            "id": f"activity_{block['day']}",
                            "type": "activity_day",
                            "label": f"{block['day']}の活動",
                            "day": block["day"],
                            "rows": block["rows"],
                            "columns": block["columns"],
                            "version": version,
                            "template_page_index": page_index,
                            "local_page_index": local_page_index,
                            "rect": block["rect"],
                            "context_rect": block["rect"],
                            "table_rect": block["table_rect"],
                        }
                    )
                )

        pages.append(
            LayoutPage.model_validate(
                {
                    "template_page_index": page_index,
                    "version": version,
                    "local_page_index": local_page_index,
                    "page_width_pt": page_w,
                    "page_height_pt": page_h,
                    "items": items,
                }
            )
        )

    doc.close()
    return Layout.model_validate(
        {
            "unit": "pdf_points",
            "coordinate_origin": "top_left",
            "versions": {
                v: {
                    "template_page_start": base,
                    "template_pages": list(range(base, base + 4)),
                }
                for v, base in VERSION_BASE_PAGE.items()
            },
            "respondent_id_rect": [35.0, 615.0, 330.0, 720.0],
            "schema_data": schema.model_dump(mode="json"),
            "pages": pages,
        }
    )


def find_layout_page(
    layout: dict[str, Any], template_page_index: int
) -> dict[str, Any]:
    for p in layout["pages"]:
        if int(p["template_page_index"]) == template_page_index:
            return p
    raise KeyError(template_page_index)
