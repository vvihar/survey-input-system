from __future__ import annotations

import argparse
from pathlib import Path

from survey_pipeline.common import read_json
from survey_pipeline.review_dataset import make_review_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answered-pdf", type=Path, required=True)
    ap.add_argument("--template-pdf", type=Path, required=True)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    manifest = make_review_dataset(
        answered_pdf=args.answered_pdf,
        template_pdf=args.template_pdf,
        layout=read_json(args.layout),
        initial_answers=read_json(args.answers),
        workdir=args.workdir,
        dpi=args.dpi,
    )
    print(f"wrote {args.workdir} ({manifest['n_items']} items)")


if __name__ == "__main__":
    main()
