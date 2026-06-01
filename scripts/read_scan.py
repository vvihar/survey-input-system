from __future__ import annotations

import argparse
from pathlib import Path

from survey_pipeline.common import read_json, write_json
from survey_pipeline.scan import read_scanned_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answered-pdf", type=Path, required=True)
    ap.add_argument("--template-pdf", type=Path, required=True)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--no-id-ocr", action="store_true")
    args = ap.parse_args()

    layout = read_json(args.layout)
    result = read_scanned_pdf(
        answered_pdf=args.answered_pdf,
        template_pdf=args.template_pdf,
        layout=layout,
        model_path=args.model,
        dpi=args.dpi,
        use_id_ocr=not args.no_id_ocr,
    )
    write_json(args.out, result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
