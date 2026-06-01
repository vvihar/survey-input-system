from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import TexActivityItem, TexDigitQuestion, TexScenarioItem, TexSchema

def _remove_tex_comments(text: str) -> str:
    # Remove comments introduced by unescaped %. Keep line breaks.
    out_lines = []
    for line in text.splitlines():
        i = 0
        cut = len(line)
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
            i += 1
        out_lines.append(line[:cut])
    return "\n".join(out_lines)

# TeXの質問環境から取れるメタデータを使う。
# レイアウト上の矩形との対応は、PDF上の検出順で付与する。

def _strip_tex(s: str) -> str:
    s = re.sub(r"%.*", "", s)
    s = s.replace("\\\\", " ")
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _read_balanced_brace(text: str, open_pos: int) -> tuple[str, int]:
    """open_pos must point to '{'. Return content and position after matching '}'."""
    if open_pos >= len(text) or text[open_pos] != "{":
        raise ValueError("expected opening brace")
    depth = 0
    start = open_pos + 1
    i = open_pos
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    raise ValueError("unbalanced braces")


def _find_environment_blocks(text: str, env: str) -> list[tuple[int, str, str]]:
    """Return (start_pos, title, body) for \\begin{env}{title}...\\end{env}.

    The title can contain nested braces such as \textbf{...}.
    """
    begin_pat = f"\\begin{{{env}}}"
    end_pat = f"\\end{{{env}}}"
    out: list[tuple[int, str, str]] = []
    pos = 0
    while True:
        b = text.find(begin_pat, pos)
        if b < 0:
            break
        i = b + len(begin_pat)
        if i < len(text) and text[i] == "[":
            close = text.find("]", i)
            i = close + 1
        while i < len(text) and text[i].isspace():
            i += 1
        title, after_title = _read_balanced_brace(text, i)
        e = text.find(end_pat, after_title)
        if e < 0:
            break
        body = text[after_title:e]
        out.append((b, title, body))
        pos = e + len(end_pat)
    return out


def _choices(body: str) -> list[str]:
    vals = []
    for m in re.finditer(r"\\choice\{(?P<x>.*?)\}", body, re.S):
        vals.append(_strip_tex(m.group("x")))
    return vals


def parse_digit_questions(tex_path: Path) -> list[TexDigitQuestion]:
    text = _remove_tex_comments(tex_path.read_text(encoding="utf-8"))
    cut = text.find("\\section{ドライバーとしてのシナリオ評価}")
    text_head = text[:cut] if cut >= 0 else text

    events: list[tuple[int, str, Any]] = []
    for m in re.finditer(r"\\section\{(.*?)\}", text_head, re.S):
        events.append((m.start(), "section", _strip_tex(m.group(1))))
    for pos, title, body in _find_environment_blocks(text_head, "question"):
        events.append((pos, "question", (title, body)))
    events.sort(key=lambda x: x[0])

    result: list[TexDigitQuestion] = []
    section_no = 0
    q_no = 0
    for _pos, kind, payload in events:
        if kind == "section":
            section_no += 1
            q_no = 0
            continue
        if section_no == 0:
            continue
        title_raw, body = payload
        q_no += 1
        title = _strip_tex(title_raw)
        ch = _choices(body)
        multiple = "全て" in body or "すべて" in body
        result.append(TexDigitQuestion(
            id=f"q{section_no}_{q_no}",
            label=title,
            choices=ch,
            min=1 if ch else None,
            max=len(ch) if ch else None,
            multiple=multiple,
        ))
    return result

def _iter_newcommands(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pat = re.compile(r"\\newcommand\{\\(?P<name>[A-Za-z]+\w*)\}")
    for m in pat.finditer(text):
        i = m.end()
        while i < len(text) and text[i].isspace():
            i += 1
        # optional arg count etc. is not used here.
        while i < len(text) and text[i] == "[":
            j = text.find("]", i)
            if j < 0:
                break
            i = j + 1
            while i < len(text) and text[i].isspace():
                i += 1
        if i < len(text) and text[i] == "{":
            try:
                body, _after = _read_balanced_brace(text, i)
                out.append((m.group("name"), body))
            except ValueError:
                pass
    return out


def parse_scenario_sets(tex_path: Path) -> dict[str, list[TexScenarioItem]]:
    text = _remove_tex_comments(tex_path.read_text(encoding="utf-8"))
    commands = dict(_iter_newcommands(text))

    scenario_bodies: dict[str, str] = {}
    for name, body in commands.items():
        if re.match(r"^(commute|ondemand|passenger)Scenario\w+$", name):
            scenario_bodies[name] = _strip_tex(body)

    sets: dict[str, list[str]] = {}
    for name, body in commands.items():
        if re.match(r"^(commute|ondemand|passenger)Set[ABC]$", name):
            sets[name] = re.findall(r"\\((?:commute|ondemand|passenger)Scenario\w+)", body)

    by_version: dict[str, list[TexScenarioItem]] = {}
    for version in ["A", "B", "C"]:
        items: list[TexScenarioItem] = []
        for part, set_prefix, q_prefix, label in [
            ("driver_commute", "commuteSet", "q4_commute", "ドライバー・通勤中"),
            ("driver_home", "ondemandSet", "q4_home", "ドライバー・自宅"),
            ("passenger", "passengerSet", "q5_passenger", "乗客"),
        ]:
            refs = sets.get(f"{set_prefix}{version}", [])
            for i, ref in enumerate(refs, start=1):
                items.append(TexScenarioItem(
                    id=f"{q_prefix}_{i}",
                    part=part,
                    label=f"{label} {i}",
                    scenario_macro=ref,
                    scenario_text=scenario_bodies.get(ref, ""),
                ))
        by_version[version] = items
    return by_version

def activity_schema() -> list[TexActivityItem]:
    days = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
    cols = ["dep_time", "dep_place", "arr_time", "arr_place", "purpose", "mode", "tolerance_min"]
    return [TexActivityItem(day=d, columns=cols) for d in days]


def build_tex_schema(tex_path: Path) -> TexSchema:
    return TexSchema(
        digit_questions=parse_digit_questions(tex_path),
        scenario_sets=parse_scenario_sets(tex_path),
        activities=activity_schema(),
    )
