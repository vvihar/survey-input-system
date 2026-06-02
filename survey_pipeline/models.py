from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
)

type Version = Literal["A", "B", "C"]
VERSION_ADAPTER: TypeAdapter[Version] = TypeAdapter(Version)


class LayoutItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = None
    type: str
    label: str = ""
    choices: list[str] = Field(default_factory=list)
    part: str | None = None
    version: Version
    template_page_index: int
    local_page_index: int
    rect: list[float] | None = None
    context_rect: list[float] | None = None
    table_rect: list[float] | None = None
    cells: list[list[float]] | None = None
    context_text: str | None = None
    day: str | None = None
    rows: int | None = None
    columns: list[str] | None = None
    scenario_macro: str | None = None
    scenario_text: str | None = None
    min: int | None = None
    max: int | None = None
    multiple: bool | None = None


class LayoutPage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template_page_index: int
    version: Literal["A", "B", "C"]
    local_page_index: int
    page_width_pt: float
    page_height_pt: float
    items: list[LayoutItem] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_id: str | None = None
    type: str
    pred_value: str | None = None
    pred_confidence: float | None = None
    ink_ratio: float | None = None
    status: str


class BookletMeta(BaseModel):
    booklet_index: int
    answered_page_start: int
    respondent_id: str
    version: Literal["A", "B", "C"]
    version_scores: dict[str, float] | None = None
    alignments: list[dict[str, Any]] = Field(default_factory=list)
    answers: dict[str, AnswerRecord] = Field(default_factory=dict)
    source_pdf: str | None = None

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value not in {"A", "B", "C"}:
            raise ValueError("version must be A, B, or C")
        return value


class InitialAnswers(BaseModel):
    source_pdf: str | list[str] | None = None
    template_pdf: str | None = None
    dpi: int = 220
    booklets: list[BookletMeta] = Field(default_factory=list)


class Manifest(BaseModel):
    answered_pdf: str | list[str]
    template_pdf: str
    dpi: int = 220
    n_items: int
    review_items: str = "review_items.jsonl"


class ReviewActivityRow(BaseModel):
    row_no: int
    dep_time: str = ""
    dep_place: str = ""
    arr_time: str = ""
    arr_place: str = ""
    purpose: str = ""
    mode: str = ""
    tolerance_min: str = ""


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_uid: str
    booklet_index: int
    respondent_id: str
    version: str
    source_pdf: str = ""
    answered_page_index: int
    template_page_index: int
    local_page_index: int
    field_id: str
    type: str
    label: str = ""
    min: int | None = None
    max: int | None = None
    multiple: bool = False
    day: str | None = None
    rows: int | None = None
    columns: list[str] | None = None
    scenario_macro: str | None = None
    scenario_text: str | None = None
    crop_path: str = ""
    context_path: str = ""
    page_image_path: str = ""
    page_width_px: int | None = None
    page_height_px: int | None = None
    page_width_pt: float | None = None
    page_height_pt: float | None = None
    rect: list[float] | None = None
    context_rect: list[float] | None = None
    table_rect: list[float] | None = None
    pred_value: str | None = None
    pred_confidence: float | None = None
    value: str | None = None
    status: str = ""
    ecc: float | None = None
    meta_respondent_id_path: str | None = None
    meta_version_path: str | None = None
    activity_rows: list[ReviewActivityRow] = Field(default_factory=list)


class ReviewItemUpdate(BaseModel):
    value: str | None = None
    status: str | None = None
    respondent_id: str | None = None
    version: str | None = None
    activity_rows: list[ReviewActivityRow] | None = None


class LayoutPageRef(BaseModel):
    template_page_index: int
    version: str
    local_page_index: int
    page_width_pt: float
    page_height_pt: float
    items: list[LayoutItem] = Field(default_factory=list)


class LayoutMetadata(BaseModel):
    unit: str = "pdf_points"
    coordinate_origin: str = "top_left"
    versions: dict[str, dict[str, object]]
    respondent_id_rect: list[float]
    schema_data: dict[str, object]


class TexDigitQuestion(BaseModel):
    id: str
    type: str = "digit"
    label: str
    choices: list[str] = Field(default_factory=list)
    min: int | None = None
    max: int | None = None
    multiple: bool = False


class TexScenarioItem(BaseModel):
    id: str
    type: str = "rating5"
    part: str
    label: str
    scenario_macro: str
    scenario_text: str
    min: int = 1
    max: int = 5


class TexActivityItem(BaseModel):
    day: str
    rows: int = 7
    columns: list[str]


class TexSchema(BaseModel):
    digit_questions: list[TexDigitQuestion]
    scenario_sets: dict[str, list[TexScenarioItem]]
    activities: list[TexActivityItem]


class LayoutDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str = "pdf_points"
    coordinate_origin: str = "top_left"
    versions: dict[str, dict[str, object]]
    respondent_id_rect: list[float]
    schema_data: TexSchema
    pages: list[LayoutPage] = Field(default_factory=list)


class Layout(LayoutDocument):
    pass


class ReadScanArgs(BaseModel):
    answered_pdfs: list[Path] = Field(default_factory=list)
    answered_dir: Path | None = None
    template_pdf: Path
    layout: Path
    model: Path | None = None
    out: Path
    dpi: int = 220
    no_id_ocr: bool = False


class ReviewDatasetArgs(BaseModel):
    answered_pdfs: list[Path] = Field(default_factory=list)
    answered_dir: Path | None = None
    template_pdf: Path
    layout: Path
    answers: Path
    workdir: Path
    dpi: int = 220
