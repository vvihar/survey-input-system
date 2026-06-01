from __future__ import annotations

import argparse
from pathlib import Path

from survey_pipeline.common import write_json
from survey_pipeline.pdf_layout import build_layout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-pdf", type=Path, required=True)
    ap.add_argument("--tex", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    layout = build_layout(args.template_pdf, args.tex)
    write_json(args.out, layout)
    n_items = sum(len(p.get("items", [])) for p in layout["pages"])
    print(f"wrote {args.out} ({n_items} items)")


if __name__ == "__main__":
    main()
