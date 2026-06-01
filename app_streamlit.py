from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from survey_pipeline.common import (
    DEFAULT_RESPONDENT_ID_RECT_PT,
    VERSION_LABEL_SEARCH_RECT_PT,
    VERSIONS,
    crop_by_rect_pt,
    iter_jsonl,
    read_json,
    render_pdf_page,
    write_jsonl,
)
from survey_pipeline.review_dataset import update_booklet_meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--workdir", type=Path, default=Path("review_work"))
    args, _ = ap.parse_known_args()
    return args


def alphabet_to_kana(s: str) -> str:
    """アルファベットをカタカナに変換。A->エー、B->ビー、..."""
    mapping = {
        "A": "エー",
        "B": "ビー",
        "C": "シー",
        "D": "ディー",
        "E": "イー",
        "F": "エフ",
        "G": "ジー",
        "H": "エイチ",
        "I": "アイ",
        "J": "ジェー",
        "K": "ケー",
        "L": "エル",
        "M": "エム",
        "N": "エヌ",
        "O": "オー",
        "P": "ピー",
        "Q": "キュー",
        "R": "アール",
        "S": "エス",
        "T": "ティー",
        "U": "ユー",
        "V": "ブイ",
        "W": "ダブリュー",
        "X": "エックス",
        "Y": "ワイ",
        "Z": "ゼット",
        "0": "ゼロ",
        "1": "イチ",
        "2": "ニ",
        "3": "サン",
        "4": "ヨン",
        "5": "ゴ",
        "6": "ロク",
        "7": "ナナ",
        "8": "ハチ",
        "9": "キュウ",
    }
    return " ".join(mapping.get(c, c) for c in s)


def load_items(workdir: Path) -> list[dict[str, Any]]:
    path = workdir / "review_items.jsonl"
    if not path.exists():
        st.error(f"not found: {path}")
        st.stop()
    return list(iter_jsonl(path))


def load_manifest(workdir: Path) -> dict[str, Any]:
    path = workdir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_initial_answers(workdir: Path) -> dict[str, Any]:
    path = workdir / "initial_answers.json"
    if not path.exists():
        return {}
    return read_json(path)


def save_items(workdir: Path, items: list[dict[str, Any]]) -> None:
    write_jsonl(workdir / "review_items.jsonl", items)


def save_manifest(workdir: Path, manifest: dict[str, Any]) -> None:
    (workdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_initial_answers(workdir: Path, initial_answers: dict[str, Any]) -> None:
    (workdir / "initial_answers.json").write_text(
        json.dumps(initial_answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def item_image_path(workdir: Path, item: dict[str, Any], context: bool = False) -> Path:
    key = "context_path" if context else "crop_path"
    return workdir / item[key]


def validate_value(value: str, item: dict[str, Any]) -> tuple[bool, str]:
    if value in ("", "?", "NA"):
        return True, "blank_or_unknown"
    if item.get("type") == "rating5":
        if value.isdigit() and 1 <= int(value) <= 5:
            return True, "ok"
        return False, "5段階評価は1〜5で入力してください。"
    if item.get("type") == "digit":
        if item.get("multiple"):
            xs = [ch for ch in value if ch.isdigit()]
            lo, hi = item.get("min"), item.get("max")
            if lo is None or hi is None or all(lo <= int(ch) <= hi for ch in xs):
                return True, "ok"
            return False, f"許容範囲は{lo}〜{hi}です。"
        if not value.isdigit():
            return False, "数字で入力してください。"
        lo, hi = item.get("min"), item.get("max")
        v = int(value)
        if lo is not None and v < lo:
            return False, f"最小値は{lo}です。"
        if hi is not None and v > hi:
            return False, f"最大値は{hi}です。"
    return True, "ok"


def sidebar_filters(items: list[dict[str, Any]]) -> list[int]:
    st.sidebar.header("絞り込み")

    source_pdfs = sorted(
        {str(x.get("source_pdf", "")) for x in items if x.get("source_pdf")}
    )
    if source_pdfs:
        selected_sources = st.sidebar.multiselect(
            "PDF", source_pdfs, default=source_pdfs, key="filter_source"
        )
    else:
        selected_sources = None

    type_options = sorted({x["type"] for x in items})
    selected_types = st.sidebar.multiselect(
        "type", type_options, default=type_options, key="filter_type"
    )
    statuses = sorted({str(x.get("status", "")) for x in items})
    selected_statuses = st.sidebar.multiselect(
        "status", statuses, default=statuses, key="filter_status"
    )
    respondent = st.sidebar.text_input("respondent_id 検索", key="filter_rid")
    only_unconfirmed = st.sidebar.checkbox(
        "未確定だけ", value=False, key="filter_unconfirmed"
    )

    idxs = []
    for i, x in enumerate(items):
        if (
            selected_sources is not None
            and str(x.get("source_pdf", "")) not in selected_sources
        ):
            continue
        if x["type"] not in selected_types:
            continue
        if str(x.get("status", "")) not in selected_statuses:
            continue
        if respondent and respondent not in x.get("respondent_id", ""):
            continue
        if only_unconfirmed and x.get("status") == "confirmed":
            continue
        idxs.append(i)
    return idxs


def apply_booklet_meta(
    items: list[dict[str, Any]], booklet_index: int, respondent_id: str, version: str
) -> None:
    for x in items:
        if int(x.get("booklet_index", -1)) == booklet_index:
            if respondent_id:
                x["respondent_id"] = respondent_id
            if version in VERSIONS:
                x["version"] = version


def booklet_editor(workdir: Path, items: list[dict[str, Any]]) -> None:
    st.sidebar.divider()
    st.sidebar.header("冊子メタ情報")
    booklets: dict[int, dict[str, Any]] = {}
    for x in items:
        bi = int(x.get("booklet_index", -1))
        if bi >= 0 and bi not in booklets:
            booklets[bi] = {
                "booklet_index": bi,
                "respondent_id": x.get("respondent_id", ""),
                "version": x.get("version", "A"),
                "source_pdf": x.get("source_pdf", ""),
                "n_items": 0,
                "confirmed": 0,
            }
        if bi in booklets:
            booklets[bi]["n_items"] += 1
            if x.get("status") == "confirmed":
                booklets[bi]["confirmed"] += 1

    if not booklets:
        return

    booklet_first_page: dict[int, int] = {}
    for x in items:
        bi = int(x.get("booklet_index", -1))
        if bi >= 0 and bi not in booklet_first_page:
            booklet_first_page[bi] = int(x.get("answered_page_index", 0))

    for bi in sorted(booklets):
        b = booklets[bi]
        with st.sidebar.expander(f"冊子 #{bi} ({b['respondent_id']})", expanded=False):
            st.caption(f"PDF: {Path(b['source_pdf']).name}")
            st.caption(f"進捗: {b['confirmed']}/{b['n_items']}")

            first_item = next(
                (x for x in items if int(x.get("booklet_index", -1)) == bi), None
            )
            rid_rel = first_item.get("meta_respondent_id_path") if first_item else None
            ver_rel = first_item.get("meta_version_path") if first_item else None
            if rid_rel:
                st.caption("調査票ID")
                st.image(str(workdir / rid_rel), use_container_width=True)
            new_rid = st.text_input(
                "respondent_id", value=b["respondent_id"], key=f"bm_rid_{bi}"
            )
            if new_rid:
                st.caption(alphabet_to_kana(new_rid))

            if ver_rel:
                st.caption("版")
                st.image(str(workdir / ver_rel), use_container_width=True)
            new_ver = st.selectbox(
                "version",
                VERSIONS,
                index=(VERSIONS.index(b["version"]) if b["version"] in VERSIONS else 0),
                key=f"bm_ver_{bi}",
            )
            if st.button("適用", key=f"bm_apply_{bi}"):
                apply_booklet_meta(items, bi, new_rid, new_ver)
                st.success(f"冊子 #{bi} を更新しました")


def rebuild_from_pdf_inputs(
    answered_pdfs: list[Path],
    template_pdf: Path,
    layout: dict[str, Any],
    answers_dir: Path,
    workdir: Path,
    dpi: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from survey_pipeline.review_dataset import make_review_datasets
    from survey_pipeline.scan import read_scanned_pdfs

    initial_answers = read_scanned_pdfs(answered_pdfs, template_pdf, layout, dpi=dpi)
    manifest = make_review_datasets(
        answered_pdfs, template_pdf, layout, initial_answers, workdir, dpi=dpi
    )
    return manifest, initial_answers


def edit_simple_item(workdir: Path, item: dict[str, Any]) -> dict[str, Any]:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.image(
            str(item_image_path(workdir, item, context=True)),
            caption="context crop",
        )
        with st.expander("回答欄だけ"):
            st.image(str(item_image_path(workdir, item, context=False)))
    with c2:
        st.write(f"**{item['field_id']}** / {item.get('label', '')}")
        st.caption(
            f"respondent={item['respondent_id']}  "
            f"version={item['version']}  "
            f"page={item['local_page_index'] + 1}  "
            f"PDF={Path(item.get('source_pdf', '')).name}"
        )
        if item.get("scenario_text"):
            st.text_area("scenario", item["scenario_text"], height=120, disabled=True)
        if item.get("type") == "digit":
            help_text = (
                f"range: {item.get('min')}〜{item.get('max')}, "
                f"multiple={item.get('multiple')}"
            )
        else:
            help_text = "rating5: 1〜5"
        st.caption(help_text)
        if item.get("pred_value") not in ("", None):
            st.write(
                f"予測値: `{item.get('pred_value')}`  /  "
                f"conf: `{item.get('pred_confidence')}`"
            )
        else:
            st.write("予測失敗")
        value_key = f"val_{item['item_uid']}"
        save_next_key = f"save_next_{item['item_uid']}"
        st.session_state.setdefault(
            value_key, "" if item.get("value") is None else str(item.get("value"))
        )
        st.session_state.setdefault(save_next_key, False)

        def _save_and_advance() -> None:
            st.session_state[save_next_key] = True

        value = st.text_input(
            "value",
            value=st.session_state[value_key],
            key=value_key,
            on_change=_save_and_advance,
        )
        ok, msg = validate_value(value, item)
        if not ok:
            st.error(msg)
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("確定", key=f"confirm_{item['item_uid']}", disabled=not ok):
            item["value"] = value
            item["status"] = "confirmed"
            st.success("confirmed")
        if col_b.button("空欄", key=f"blank_{item['item_uid']}"):
            item["value"] = ""
            item["status"] = "blank"
            st.success("blank")
        if col_c.button("判読不能", key=f"unk_{item['item_uid']}"):
            item["value"] = "?"
            item["status"] = "unknown"
            st.success("unknown")

        if st.session_state.get(save_next_key) and ok:
            item["value"] = value
            item["status"] = "confirmed"
            save_items(workdir, st.session_state["review_items"])
            save_manifest(workdir, st.session_state.get("manifest", {}))
            st.session_state[save_next_key] = False
            st.session_state["current_pos"] = min(
                st.session_state["current_pos"] + 1,
                st.session_state.get("current_pos", 0) + 1,
            )
            st.rerun()
    return item


def edit_activity_item(workdir: Path, item: dict[str, Any]) -> dict[str, Any]:
    st.image(
        str(item_image_path(workdir, item, context=True)),
        caption=f"{item.get('day')} activity table",
    )
    rows = item.get("activity_rows") or []
    df = pd.DataFrame(rows)
    ordered_cols = [
        "row_no",
        "dep_time",
        "dep_place",
        "arr_time",
        "arr_place",
        "purpose",
        "mode",
        "tolerance_min",
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = ""
    edited = st.data_editor(
        df[ordered_cols],
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        key=f"activity_{item['item_uid']}",
    )
    if st.button("活動表を保存", key=f"save_act_{item['item_uid']}"):
        item["activity_rows"] = edited.to_dict(orient="records")
        item["status"] = "confirmed"
        st.success("saved")
    return item


def autofocus_input(input_key: str) -> None:
    components.html(
        f"""
        <script>
          const key = {json.dumps(input_key)};
          const focus = () => {{
            const el = window.parent.document.querySelector(`[data-testid="stTextInput"] input[key="${{key}}"]`);
            if (el) {{
              el.focus();
              el.select?.();
              return true;
            }}
            return false;
          }};
          let n = 0;
          const timer = setInterval(() => {{
            n += 1;
            if (focus() || n > 25) clearInterval(timer);
          }}, 80);
        </script>
        """,
        height=0,
    )


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    st.set_page_config(page_title="Survey review", layout="wide")
    st.title("調査票入力システム")
    st.caption(str(workdir.resolve()))

    if "review_items" not in st.session_state:
        st.session_state["review_items"] = load_items(workdir)
        st.session_state["current_pos"] = 0
        st.session_state["manifest"] = load_manifest(workdir)
        st.session_state["initial_answers"] = load_initial_answers(workdir)

    items: list[dict[str, Any]] = st.session_state["review_items"]
    if not items:
        st.error("項目がありません。")
        return

    idxs = sidebar_filters(items)
    booklet_editor(workdir, items)

    if not idxs:
        st.warning("該当する項目がありません。")
        return
    st.sidebar.write(f"{len(idxs)} / {len(items)} items")

    max_pos = len(idxs) - 1
    st.session_state["current_pos"] = min(st.session_state["current_pos"], max_pos)
    pos = st.sidebar.number_input(
        "position",
        min_value=0,
        max_value=max_pos,
        value=st.session_state["current_pos"],
    )
    st.session_state["current_pos"] = int(pos)
    idx = idxs[st.session_state["current_pos"]]
    item = items[idx]

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("← 前"):
        st.session_state["current_pos"] = max(0, st.session_state["current_pos"] - 1)
        st.rerun()
    if b2.button("次 →"):
        st.session_state["current_pos"] = min(
            max_pos, st.session_state["current_pos"] + 1
        )
        st.rerun()
    if b3.button("保存"):
        save_items(workdir, items)
        save_manifest(workdir, st.session_state.get("manifest", {}))
        st.success("saved")
    if b4.button("保存して次"):
        save_items(workdir, items)
        save_manifest(workdir, st.session_state.get("manifest", {}))
        st.session_state["current_pos"] = min(
            max_pos, st.session_state["current_pos"] + 1
        )
        st.rerun()

    with st.expander("現在の冊子メタ情報", expanded=False):
        current_bi = int(item.get("booklet_index", 0))
        cur_rid = st.text_input(
            "respondent_id",
            value=str(item.get("respondent_id", "")),
            key=f"current_rid_{current_bi}",
        )
        cur_ver = st.selectbox(
            "version",
            VERSIONS,
            index=(
                VERSIONS.index(item.get("version", "A"))
                if item.get("version") in VERSIONS
                else 0
            ),
            key=f"current_ver_{current_bi}",
        )
        if st.button("今の冊子に反映", key=f"current_apply_{current_bi}"):
            apply_booklet_meta(items, current_bi, cur_rid, cur_ver)
            ia = st.session_state.get("initial_answers", {})
            if ia:
                st.session_state["initial_answers"] = update_booklet_meta(
                    ia, current_bi, cur_rid, cur_ver
                )
                save_initial_answers(workdir, st.session_state["initial_answers"])
            st.success(f"冊子 #{current_bi} を更新しました", icon="✅")

    st.divider()
    if item["type"] in ("digit", "rating5"):
        items[idx] = edit_simple_item(workdir, item)
        autofocus_input(f"val_{item['item_uid']}")
    elif item["type"] == "activity_day":
        items[idx] = edit_activity_item(workdir, item)
    else:
        st.json(item)

    st.divider()
    with st.expander("raw item"):
        st.code(json.dumps(items[idx], ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
