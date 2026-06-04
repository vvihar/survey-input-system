from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from scripts.make_review_dataset import collect_pdfs
from survey_pipeline.common import read_layout, write_json
from survey_pipeline.pdf_layout import build_layout
from survey_pipeline.review_dataset import (
    make_review_dataset,
    make_review_datasets,
    source_pdf_key,
)
from survey_pipeline.scan import read_scanned_pdf, read_scanned_pdfs

app = typer.Typer(add_completion=False)


def ensure_layout(template_pdf: Path, tex: Path, layout: Path) -> dict[str, Any]:
    if not layout.exists():
        typer.echo(f"building layout: {layout}")
        write_json(layout, build_layout(template_pdf, tex))
    return read_layout(layout).model_dump(mode="json")


@app.command()
def prepare(
    answered_pdf: list[Path] = typer.Option([], "--answered-pdf", "-p"),
    answered_dir: Path | None = typer.Option(None, "--answered-dir", "-d"),
    template_pdf: Path = typer.Option(Path("survey-sheet.pdf"), "--template-pdf"),
    tex: Path = typer.Option(Path("survey-sheet.tex"), "--tex"),
    model: Path | None = typer.Option(Path("models/mnist.pt"), "--model"),
    out_dir: Path = typer.Option(Path("outputs"), "--out-dir", "-o"),
    dpi: int = typer.Option(220, "--dpi"),
    no_id_ocr: bool = typer.Option(False, "--no-id-ocr"),
    rebuild_layout: bool = typer.Option(False, "--rebuild-layout"),
) -> None:
    pdf_inputs = list(answered_pdf)
    if answered_dir is not None:
        pdf_inputs.append(answered_dir)
    pdfs = sorted(set(collect_pdfs(pdf_inputs)))
    if not pdfs:
        raise typer.BadParameter("specify --answered-pdf or --answered-dir")

    layout_path = out_dir / "layout_survey.json"
    if rebuild_layout and layout_path.exists():
        layout_path.unlink()
    layout = ensure_layout(template_pdf, tex, layout_path)

    answers_dir = out_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    if len(pdfs) == 1:
        pdf = pdfs[0]
        pdf_key = source_pdf_key(pdf)
        answers_path = answers_dir / f"{pdf_key}.json"
        review_dir = out_dir / "review" / pdf_key
        typer.echo(f"scanning: {pdf.name}")
        result = read_scanned_pdf(
            answered_pdf=pdf,
            template_pdf=template_pdf,
            layout=layout,
            model_path=model if model is not None and model.exists() else None,
            dpi=dpi,
            use_id_ocr=not no_id_ocr,
        )
        answers = result.model_dump(mode="json")
        write_json(answers_path, answers)
        typer.echo(f"building review dataset: {pdf.name}")
        manifest = make_review_dataset(
            answered_pdf=pdf,
            template_pdf=template_pdf,
            layout=layout,
            initial_answers=answers,
            workdir=review_dir,
            dpi=dpi,
        )
        typer.echo(f"answers: {answers_path}")
        typer.echo(f"review:  {review_dir} ({manifest['n_items']} items)")
        return

    answers = read_scanned_pdfs(
        answered_pdfs=pdfs,
        template_pdf=template_pdf,
        layout=layout,
        model_path=model if model is not None and model.exists() else None,
        dpi=dpi,
        use_id_ocr=not no_id_ocr,
    ).model_dump(mode="json")
    answers_path = answers_dir / "merged.json"
    review_dir = out_dir / "review_merged"
    write_json(answers_path, answers)
    typer.echo("building review dataset: merged PDFs")
    manifest = make_review_datasets(
        answered_pdfs=pdfs,
        template_pdf=template_pdf,
        layout=layout,
        merged_initial_answers=answers,
        workdir=review_dir,
        dpi=dpi,
    )
    typer.echo(f"answers: {answers_path}")
    typer.echo(f"review:  {review_dir} ({manifest['n_items']} items)")


@app.command()
def layout(
    template_pdf: Path = typer.Option(Path("survey-sheet.pdf"), "--template-pdf"),
    tex: Path = typer.Option(Path("survey-sheet.tex"), "--tex"),
    out: Path = typer.Option(Path("outputs/layout_survey.json"), "--out", "-o"),
) -> None:
    write_json(out, build_layout(template_pdf, tex))
    typer.echo(f"layout: {out}")


if __name__ == "__main__":
    app()
