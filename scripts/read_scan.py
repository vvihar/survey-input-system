from __future__ import annotations

from pathlib import Path

import typer

from survey_pipeline.common import read_layout, write_json
from survey_pipeline.models import ReadScanArgs
from survey_pipeline.scan import read_scanned_pdf, read_scanned_pdfs

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
    answered_pdf: list[Path] = typer.Option([], "--answered-pdf", help="answered PDF file (repeatable)"),
    answered_dir: Path | None = typer.Option(None, "--answered-dir", help="directory containing answered PDFs"),
    template_pdf: Path = typer.Option(..., "--template-pdf"),
    layout: Path = typer.Option(..., "--layout"),
    model: Path | None = typer.Option(None, "--model"),
    out: Path = typer.Option(..., "--out"),
    dpi: int = typer.Option(220, "--dpi"),
    no_id_ocr: bool = typer.Option(False, "--no-id-ocr"),
) -> None:
    args = ReadScanArgs(
        answered_pdfs=answered_pdf,
        answered_dir=answered_dir,
        template_pdf=template_pdf,
        layout=layout,
        model=model,
        out=out,
        dpi=dpi,
        no_id_ocr=no_id_ocr,
    )
    pdfs = []
    if args.answered_dir is not None:
        pdfs.extend(collect_pdfs([args.answered_dir]))
    if args.answered_pdfs:
        pdfs.extend(collect_pdfs(args.answered_pdfs))
    pdfs = sorted(set(pdfs))
    if not pdfs:
        raise typer.BadParameter("specify at least one --answered-pdf or --answered-dir")

    layout_model = read_layout(args.layout).model_dump()

    if len(pdfs) == 1:
        result = read_scanned_pdf(
            answered_pdf=pdfs[0],
            template_pdf=args.template_pdf,
            layout=layout_model,
            model_path=args.model,
            dpi=args.dpi,
            use_id_ocr=not args.no_id_ocr,
        )
    else:
        result = read_scanned_pdfs(
            answered_pdfs=pdfs,
            template_pdf=args.template_pdf,
            layout=layout_model,
            model_path=args.model,
            dpi=args.dpi,
            use_id_ocr=not args.no_id_ocr,
        )

    result_data = result.model_dump(mode="json")
    write_json(args.out, result_data)
    typer.echo(f"wrote {args.out} ({len(result_data['booklets'])} booklets from {len(pdfs)} PDF(s))")


if __name__ == "__main__":
    app()
