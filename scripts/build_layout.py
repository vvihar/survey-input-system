from __future__ import annotations

from pathlib import Path

import typer

from survey_pipeline.common import write_json
from survey_pipeline.pdf_layout import build_layout

app = typer.Typer(add_completion=False)


@app.command()
def main(
    template_pdf: Path = typer.Option(..., "--template-pdf"),
    tex: Path = typer.Option(..., "--tex"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    layout = build_layout(template_pdf, tex)
    write_json(out, layout)
    n_items = sum(len(p.items) for p in layout.pages)
    typer.echo(f"wrote {out} ({n_items} items)")


if __name__ == "__main__":
    app()
