from __future__ import annotations

from pathlib import Path

import typer

from survey_pipeline.common import read_initial_answers, read_layout
from survey_pipeline.review_dataset import make_review_dataset, source_pdf_key

app = typer.Typer(add_completion=False)


def collect_pdfs(inputs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in inputs:
        if p.is_dir():
            out.extend(sorted(p.glob("**/*.pdf")))
        elif p.suffix.lower() == ".pdf":
            out.append(p)
    return out


@app.command()
def main(
    answered_pdfs: list[Path] = typer.Option([], "--answered-pdfs"),
    answered_dir: Path | None = typer.Option(None, "--answered-dir"),
    template_pdf: Path = typer.Option(..., "--template-pdf"),
    layout: Path = typer.Option(..., "--layout"),
    answers_dir: Path = typer.Option(..., "--answers-dir"),
    workdir: Path = typer.Option(..., "--workdir"),
    dpi: int = typer.Option(220, "--dpi"),
) -> None:
    inputs = list(answered_pdfs)
    if answered_dir is not None:
        inputs.append(answered_dir)
    pdfs = collect_pdfs(inputs)
    workdir.mkdir(parents=True, exist_ok=True)

    layout_model = read_layout(layout).model_dump()
    for pdf in pdfs:
        pdf_key = source_pdf_key(pdf)
        answers_path = answers_dir / f"{pdf_key}.json"
        if not answers_path.exists():
            answers_path = answers_dir / f"{pdf.stem}.json"
        if not answers_path.exists():
            continue
        out = workdir / pdf_key
        manifest = make_review_dataset(
            answered_pdf=pdf,
            template_pdf=template_pdf,
            layout=layout_model,
            initial_answers=read_initial_answers(answers_path).model_dump(mode="json"),
            workdir=out,
            dpi=dpi,
        )
        typer.echo(f"wrote {out} ({manifest['n_items']} items)")


if __name__ == "__main__":
    app()
