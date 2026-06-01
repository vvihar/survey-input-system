from __future__ import annotations

from pathlib import Path

import typer

from survey_pipeline.common import read_initial_answers, read_layout
from survey_pipeline.models import ReviewDatasetArgs
from survey_pipeline.review_dataset import make_review_dataset, make_review_datasets

app = typer.Typer(add_completion=False)


def collect_pdfs(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for p in paths:
        if p.is_dir():
            result.extend(sorted(p.glob("*.pdf")))
        elif p.is_file():
            result.append(p)
    return result


@app.command()
def main(
    answered_pdf: list[Path] = typer.Option([], "--answered-pdf"),
    answered_dir: Path | None = typer.Option(None, "--answered-dir"),
    template_pdf: Path = typer.Option(..., "--template-pdf"),
    layout: Path = typer.Option(..., "--layout"),
    answers: Path = typer.Option(..., "--answers"),
    workdir: Path = typer.Option(..., "--workdir"),
    dpi: int = typer.Option(220, "--dpi"),
) -> None:
    args = ReviewDatasetArgs(
        answered_pdfs=answered_pdf,
        answered_dir=answered_dir,
        template_pdf=template_pdf,
        layout=layout,
        answers=answers,
        workdir=workdir,
        dpi=dpi,
    )
    pdfs = []
    if args.answered_dir is not None:
        pdfs.extend(collect_pdfs([args.answered_dir]))
    if args.answered_pdfs:
        pdfs.extend(collect_pdfs(args.answered_pdfs))
    pdfs = sorted(set(pdfs))

    layout_model = read_layout(args.layout).model_dump()
    answers_model = read_initial_answers(args.answers).model_dump(mode="json")

    if len(pdfs) <= 1:
        pdf = pdfs[0] if pdfs else Path(str(answers_model.get("source_pdf", answers_model.get("answered_pdf", ""))))
        manifest = make_review_dataset(
            answered_pdf=pdf,
            template_pdf=args.template_pdf,
            layout=layout_model,
            initial_answers=answers_model,
            workdir=args.workdir,
            dpi=args.dpi,
        )
    else:
        manifest = make_review_datasets(
            answered_pdfs=pdfs,
            template_pdf=args.template_pdf,
            layout=layout_model,
            merged_initial_answers=answers_model,
            workdir=args.workdir,
            dpi=args.dpi,
        )
    typer.echo(f"wrote {args.workdir} ({manifest['n_items']} items)")


if __name__ == "__main__":
    app()
