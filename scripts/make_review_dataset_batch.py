from __future__ import annotations

import argparse
from pathlib import Path

from survey_pipeline.common import read_json
from survey_pipeline.review_dataset import make_review_dataset


def collect_pdfs(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            out.extend(sorted(p.glob("**/*.pdf")))
        elif p.suffix.lower() == ".pdf":
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answered-pdfs", type=Path, nargs="*", default=[])
    ap.add_argument("--answered-dir", type=Path, default=None)
    ap.add_argument("--template-pdf", type=Path, required=True)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--answers-dir", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    inputs = list(args.answered_pdfs)
    if args.answered_dir is not None:
        inputs.append(args.answered_dir)
    pdfs = collect_pdfs(inputs)
    args.workdir.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        answers_path = args.answers_dir / f"{pdf.stem}.json"
        if not answers_path.exists():
            continue
        out = args.workdir / pdf.stem
        manifest = make_review_dataset(
            answered_pdf=pdf,
            template_pdf=args.template_pdf,
            layout=read_json(args.layout),
            initial_answers=read_json(answers_path),
            workdir=out,
            dpi=args.dpi,
        )
        print(f"wrote {out} ({manifest['n_items']} items)")


if __name__ == "__main__":
    main()
