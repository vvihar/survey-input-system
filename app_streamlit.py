from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from survey_pipeline.common import iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--workdir", type=Path, default=Path("review_work"))
    args, _ = ap.parse_known_args()
    return args


def load_items(workdir: Path) -> list[dict[str, Any]]:
    path = workdir / "review_items.jsonl"
    if not path.exists():
        st.error(f"not found: {path}")
        st.stop()
    return list(iter_jsonl(path))


def save_items(workdir: Path, items: list[dict[str, Any]]) -> None:
    write_jsonl(workdir / "review_items.jsonl", items)


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
    st.sidebar.header("Filter")
    type_options = sorted({x["type"] for x in items})
    selected_types = st.sidebar.multiselect("type", type_options, default=type_options)
    statuses = sorted({str(x.get("status", "")) for x in items})
    selected_statuses = st.sidebar.multiselect("status", statuses, default=statuses)
    respondent = st.sidebar.text_input("respondent_id contains", "")
    only_unconfirmed = st.sidebar.checkbox("未確定だけ", value=False)
    idxs = []
    for i, x in enumerate(items):
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


def edit_simple_item(workdir: Path, item: dict[str, Any]) -> dict[str, Any]:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.image(
            str(item_image_path(workdir, item, context=True)), caption="context crop"
        )
        with st.expander("回答欄だけ"):
            st.image(str(item_image_path(workdir, item, context=False)))
    with c2:
        st.write(f"**{item['field_id']}** / {item.get('label', '')}")
        st.caption(
            f"respondent={item['respondent_id']} version={item['version']} page={item['local_page_index'] + 1}"
        )
        if item.get("scenario_text"):
            st.text_area("scenario", item["scenario_text"], height=120, disabled=True)
        if item.get("type") == "digit":
            help_text = f"range: {item.get('min')}〜{item.get('max')}, multiple={item.get('multiple')}"
        else:
            help_text = "rating5: 1〜5"
        st.caption(help_text)
        if item.get("pred_value") != "":
            st.write(
                f"予測値: `{item.get('pred_value')}` / conf: `{item.get('pred_confidence')}`"
            )
        else:
            st.write("予測失敗")
        value = st.text_input(
            "value",
            value="" if item.get("value") is None else str(item.get("value")),
            key=f"val_{item['item_uid']}",
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


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    st.set_page_config(page_title="Survey review", layout="wide")
    st.title("調査票入力システム")
    st.caption(str(workdir.resolve()))

    if "review_items" not in st.session_state:
        st.session_state["review_items"] = load_items(workdir)
        st.session_state["current_pos"] = 0

    items: list[dict[str, Any]] = st.session_state["review_items"]
    idxs = sidebar_filters(items)
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
        st.success("saved")
    if b4.button("保存して次"):
        save_items(workdir, items)
        st.session_state["current_pos"] = min(
            max_pos, st.session_state["current_pos"] + 1
        )
        st.rerun()

    st.divider()
    if item["type"] in ("digit", "rating5"):
        items[idx] = edit_simple_item(workdir, item)
    elif item["type"] == "activity_day":
        items[idx] = edit_activity_item(workdir, item)
    else:
        st.json(item)

    st.divider()
    with st.expander("raw item"):
        st.code(json.dumps(items[idx], ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
