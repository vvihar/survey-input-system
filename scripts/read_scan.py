from __future__ import annotations

import argparse
from pathlib import Path

from survey_pipeline.common import read_json, write_json
from survey_pipeline.scan import read_scanned_pdf, read_scanned_pdfs


def collect_pdfs(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for p in paths:
        if p.is_dir():
            result.extend(sorted(p.glob("*.pdf")))
        elif p.is_file():
            result.append(p)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answered-pdf", type=Path, action="append", default=[], help="answered PDF file (repeatable or use --answered-dir)")
    ap.add_argument("--answered-dir", type=Path, default=None, help="directory containing answered PDFs")
    ap.add_argument("--template-pdf", type=Path, required=True)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--no-id-ocr", action="store_true")
    args = ap.parse_args()

    pdfs: list[Path] = []
    if args.answered_dir:
        pdfs.extend(collect_pdfs([args.answered_dir]))
    if args.answered_pdf:
        pdfs.extend(collect_pdfs(args.answered_pdf))
    pdfs = sorted(set(pdfs))
    if not pdfs:
        ap.error("specify at least one --answered-pdf or --answered-dir")

    layout = read_json(args.layout)

    if len(pdfs) == 1:
        result = read_scanned_pdf(
            answered_pdf=pdfs[0],
            template_pdf=args.template_pdf,
            layout=layout,
            model_path=args.model,
            dpi=args.dpi,
            use_id_ocr=not args.no_id_ocr,
        )
    else:
        result = read_scanned_pdfs(
            answered_pdfs=pdfs,
            template_pdf=args.template_pdf,
            layout=layout,
            model_path=args.model,
            dpi=args.dpi,
            use_id_ocr=not args.no_id_ocr,
        )

    write_json(args.out, result)
    n_booklets = len(result["booklets"])
    print(f"wrote {args.out} ({n_booklets} booklets from {len(pdfs)} PDF(s))")


if __name__ == "__main__":
    main()
