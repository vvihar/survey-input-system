from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import typer

from survey_pipeline.common import iter_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    workdir: Path = typer.Option(..., "--workdir"),
    out_dir: Path = typer.Option(..., "--out-dir"),
) -> None:
    rows = list(iter_jsonl(workdir / "review_items.jsonl"))
    out_dir.mkdir(parents=True, exist_ok=True)

    simple_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    for r in rows:
        if r["type"] in ("digit", "rating5"):
            simple_rows.append({
                "respondent_id": r["respondent_id"],
                "version": r["version"],
                "source_pdf": r.get("source_pdf", ""),
                "field_id": r["field_id"],
                "type": r["type"],
                "label": r.get("label"),
                "value": r.get("value"),
                "status": r.get("status"),
                "pred_value": r.get("pred_value"),
                "pred_confidence": r.get("pred_confidence"),
            })
        elif r["type"] == "activity_day":
            for ar in r.get("activity_rows", []):
                activity_rows.append({
                    "respondent_id": r["respondent_id"],
                    "version": r["version"],
                    "source_pdf": r.get("source_pdf", ""),
                    "day": r.get("day"),
                    **ar,
                })

    simple = pd.DataFrame(simple_rows)
    simple.to_csv(out_dir / "answers_long.csv", index=False)
    if not simple.empty:
        wide = simple.pivot_table(index=["respondent_id", "version", "source_pdf"], columns="field_id", values="value", aggfunc="first").reset_index()
        wide.to_csv(out_dir / "answers_wide.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "answers_wide.csv", index=False)

    pd.DataFrame(activity_rows).to_csv(out_dir / "activities.csv", index=False)
    typer.echo(f"wrote {out_dir}")


if __name__ == "__main__":
    app()
