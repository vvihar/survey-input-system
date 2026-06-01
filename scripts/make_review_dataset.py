from __future__ import annotations

import argparse
from pathlib import Path

from survey_pipeline.common import read_json
from survey_pipeline.review_dataset import make_review_dataset, make_review_datasets


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
    ap.add_argument("--answers", type=Path, required=True, help="initial_answers.json from read_scan.py")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    pdfs: list[Path] = []
    if args.answered_dir:
        pdfs.extend(collect_pdfs([args.answered_dir]))
    if args.answered_pdf:
        pdfs.extend(collect_pdfs(args.answered_pdf))
    pdfs = sorted(set(pdfs))

    layout = read_json(args.layout)
    answers = read_json(args.answers)

    if len(pdfs) <= 1:
        pdf = pdfs[0] if pdfs else Path(str(answers.get("source_pdf", answers.get("answered_pdf", ""))))
        manifest = make_review_dataset(
            answered_pdf=pdf,
            template_pdf=args.template_pdf,
            layout=layout,
            initial_answers=answers,
            workdir=args.workdir,
            dpi=args.dpi,
        )
    else:
        manifest = make_review_datasets(
            answered_pdfs=pdfs,
            template_pdf=args.template_pdf,
            layout=layout,
            merged_initial_answers=answers,
            workdir=args.workdir,
            dpi=args.dpi,
        )
    print(f"wrote {args.workdir} ({manifest['n_items']} items)")


if __name__ == "__main__":
    main()
