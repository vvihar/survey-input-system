from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from survey_pipeline.common import VERSIONS, iter_review_items, read_json, write_jsonl
from survey_pipeline.models import ReviewActivityRow, ReviewItem
from survey_pipeline.review_dataset import update_booklet_meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--workdir", type=Path, default=Path("review_work"))
    args, _ = ap.parse_known_args()
    return args


def load_items(workdir: Path) -> list[ReviewItem]:
    path = workdir / "review_items.jsonl"
    if not path.exists():
        st.error(f"not found: {path}")
        st.stop()
    return list(iter_review_items(path))


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


def save_items(workdir: Path, items: list[ReviewItem]) -> None:
    write_jsonl(
        workdir / "review_items.jsonl", [x.model_dump(mode="json") for x in items]
    )


def save_manifest(workdir: Path, manifest: dict[str, Any]) -> None:
    (workdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_initial_answers(workdir: Path, initial_answers: dict[str, Any]) -> None:
    (workdir / "initial_answers.json").write_text(
        json.dumps(initial_answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def item_image_path(workdir: Path, item: ReviewItem, context: bool = False) -> Path:
    key = "context_path" if context else "crop_path"
    return workdir / getattr(item, key)


def validate_value(value: str, item: ReviewItem) -> tuple[bool, str]:
    if value in ("", "?", "NA"):
        return True, "blank_or_unknown"
    if item.type == "rating5":
        if value.isdigit() and 1 <= int(value) <= 5:
            return True, "ok"
        return False, "5段階評価は1〜5で入力してください。"
    if item.type == "digit":
        if item.multiple:
            xs = [ch for ch in value if ch.isdigit()]
            if (
                item.min is None
                or item.max is None
                or all(item.min <= int(ch) <= item.max for ch in xs)
            ):
                return True, "ok"
            return False, f"許容範囲は{item.min}〜{item.max}です。"
        if not value.isdigit():
            return False, "数字で入力してください。"
        v = int(value)
        if item.min is not None and v < item.min:
            return False, f"最小値は{item.min}です。"
        if item.max is not None and v > item.max:
            return False, f"最大値は{item.max}です。"
    return True, "ok"


def sidebar_filters(items: list[ReviewItem]) -> list[int]:
    st.sidebar.header("絞り込み")
    source_pdfs = sorted({x.source_pdf for x in items if x.source_pdf})
    selected_sources = (
        st.sidebar.multiselect(
            "PDF", source_pdfs, default=source_pdfs, key="filter_source"
        )
        if source_pdfs
        else None
    )
    type_options = sorted({x.type for x in items})
    selected_types = st.sidebar.multiselect(
        "type", type_options, default=type_options, key="filter_type"
    )
    statuses = sorted({x.status for x in items})
    selected_statuses = st.sidebar.multiselect(
        "status", statuses, default=statuses, key="filter_status"
    )
    respondent = st.sidebar.text_input("respondent_id 検索", key="filter_rid")
    only_unconfirmed = st.sidebar.checkbox(
        "未確定だけ", value=False, key="filter_unconfirmed"
    )
    idxs: list[int] = []
    for i, x in enumerate(items):
        if selected_sources is not None and x.source_pdf not in selected_sources:
            continue
        if x.type not in selected_types:
            continue
        if x.status not in selected_statuses:
            continue
        if respondent and respondent not in x.respondent_id:
            continue
        if only_unconfirmed and x.status == "confirmed":
            continue
        idxs.append(i)
    return idxs


def apply_booklet_meta(
    items: list[ReviewItem], booklet_index: int, respondent_id: str, version: str
) -> None:
    for x in items:
        if x.booklet_index == booklet_index:
            if respondent_id:
                x.respondent_id = respondent_id
            if version in VERSIONS:
                x.version = version


def booklet_editor(workdir: Path, items: list[ReviewItem]) -> None:
    st.sidebar.divider()
    st.sidebar.header("冊子メタ情報")
    booklets: dict[int, dict[str, Any]] = {}
    for x in items:
        bi = int(x.booklet_index)
        booklets.setdefault(
            bi,
            {
                "booklet_index": bi,
                "respondent_id": x.respondent_id,
                "version": x.version,
                "source_pdf": x.source_pdf,
                "n_items": 0,
                "confirmed": 0,
            },
        )
        booklets[bi]["n_items"] += 1
        if x.status == "confirmed":
            booklets[bi]["confirmed"] += 1
    if not booklets:
        return
    for bi in sorted(booklets):
        b = booklets[bi]
        with st.sidebar.expander(f"冊子 #{bi} ({b['respondent_id']})", expanded=False):
            st.caption(f"PDF: {Path(b['source_pdf']).name}")
            st.caption(f"進捗: {b['confirmed']}/{b['n_items']}")
            first_item = next((x for x in items if x.booklet_index == bi), None)
            if first_item and first_item.meta_respondent_id_path:
                st.caption("調査票ID")
                st.image(
                    str(workdir / first_item.meta_respondent_id_path),
                    use_container_width=True,
                )
            new_rid = st.text_input(
                "respondent_id", value=b["respondent_id"], key=f"bm_rid_{bi}"
            )
            if new_rid:
                st.caption(new_rid)
            if first_item and first_item.meta_version_path:
                st.caption("版")
                st.image(
                    str(workdir / first_item.meta_version_path),
                    use_container_width=True,
                )
            new_ver = st.selectbox(
                "version",
                VERSIONS,
                index=VERSIONS.index(b["version"]) if b["version"] in VERSIONS else 0,
                key=f"bm_ver_{bi}",
            )
            if st.button("適用", key=f"bm_apply_{bi}") and new_rid:
                apply_booklet_meta(items, bi, new_rid, new_ver)
                st.success(f"冊子 #{bi} を更新しました")


def edit_simple_item(workdir: Path, item: ReviewItem) -> ReviewItem:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.image(
            str(item_image_path(workdir, item, context=True)), caption="context crop"
        )
        with st.expander("回答欄だけ"):
            st.image(str(item_image_path(workdir, item, context=False)))
    with c2:
        st.write(f"**{item.field_id}** / {item.label}")
        st.caption(
            f"respondent={item.respondent_id}  version={item.version}  page={item.local_page_index + 1}  PDF={Path(item.source_pdf).name}"
        )
        if item.scenario_text:
            st.text_area("scenario", item.scenario_text, height=120, disabled=True)
        help_text = (
            f"range: {item.min}〜{item.max}, multiple={item.multiple}"
            if item.type == "digit"
            else "rating5: 1〜5"
        )
        st.caption(help_text)
        if item.pred_value != "":
            st.write(f"予測値: `{item.pred_value}`  /  conf: `{item.pred_confidence}`")
        else:
            st.write("予測失敗")
        value_key = f"val_{item.item_uid}"
        save_next_key = f"save_next_{item.item_uid}"
        st.session_state.setdefault(
            value_key, "" if item.value is None else str(item.value)
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
        if value:
            ok, msg = validate_value(value, item)
            if not ok:
                st.error(msg)
            col_a, col_b, col_c = st.columns(3)
            if col_a.button("確定", key=f"confirm_{item.item_uid}", disabled=not ok):
                item.value = value
                item.status = "confirmed"
                st.success("confirmed")
            if col_b.button("空欄", key=f"blank_{item.item_uid}"):
                item.value = ""
                item.status = "blank"
                st.success("blank")
            if col_c.button("判読不能", key=f"unk_{item.item_uid}"):
                item.value = "?"
                item.status = "unknown"
                st.success("unknown")
            if st.session_state.get(save_next_key) and ok:
                item.value = value
                item.status = "confirmed"
                save_items(workdir, st.session_state["review_items"])
                save_manifest(workdir, st.session_state.get("manifest", {}))
                st.session_state[save_next_key] = False
                st.session_state["current_pos"] = min(
                    st.session_state["current_pos"] + 1,
                    len(st.session_state["review_items"]) - 1,
                )
                st.rerun()
    return item


def edit_activity_item(workdir: Path, item: ReviewItem) -> ReviewItem:
    st.image(
        str(item_image_path(workdir, item, context=True)),
        caption=f"{item.day} activity table",
    )
    df = pd.DataFrame([r.model_dump() for r in item.activity_rows])
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
        key=f"activity_{item.item_uid}",
    )
    if st.button("活動表を保存", key=f"save_act_{item.item_uid}"):
        item.activity_rows = [
            ReviewActivityRow.model_validate(x)
            for x in edited.to_dict(orient="records")  # pyright: ignore
        ]
        item.status = "confirmed"
        st.success("saved")
    return item


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

    items: list[ReviewItem] = st.session_state["review_items"]
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
    prev_key = f"nav_prev_{item.item_uid}"
    next_key = f"nav_next_{item.item_uid}"
    if b1.button("← 前", key=prev_key):
        st.session_state["current_pos"] = max(0, st.session_state["current_pos"] - 1)
        st.rerun()
    if b2.button("次 →", key=next_key):
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
        current_bi = int(item.booklet_index)
        cur_rid = st.text_input(
            "respondent_id", value=item.respondent_id, key=f"current_rid_{current_bi}"
        )
        cur_ver = st.selectbox(
            "version",
            VERSIONS,
            index=VERSIONS.index(item.version) if item.version in VERSIONS else 0,
            key=f"current_ver_{current_bi}",
        )
        if st.button("今の冊子に反映", key=f"current_apply_{current_bi}") and cur_rid:
            apply_booklet_meta(items, current_bi, cur_rid, cur_ver)
            ia = st.session_state.get("initial_answers", {})
            if ia:
                st.session_state["initial_answers"] = update_booklet_meta(
                    ia, current_bi, cur_rid, cur_ver
                )
                save_initial_answers(workdir, st.session_state["initial_answers"])
            st.success(f"冊子 #{current_bi} を更新しました", icon="✅")

    st.divider()
    if item.type in ("digit", "rating5"):
        items[idx] = edit_simple_item(workdir, item)
    elif item.type == "activity_day":
        items[idx] = edit_activity_item(workdir, item)
    else:
        st.json(item.model_dump(mode="json"))

    st.divider()
    with st.expander("raw item"):
        st.code(
            json.dumps(
                items[idx].model_dump(mode="json"), ensure_ascii=False, indent=2
            ),
            language="json",
        )


if __name__ == "__main__":
    main()
