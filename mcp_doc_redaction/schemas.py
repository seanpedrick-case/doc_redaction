from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BaseOptions(BaseModel):
    """Common optional fields for requests (kept small + permissive)."""

    # If caller wants to override the server output folder, they can pass it,
    # but most HF deployments write to container-local OUTPUT_FOLDER.
    output_dir: str | None = None


class RedactDocumentOptions(BaseOptions):
    ocr_method: str | None = None
    pii_method: str | None = None
    allow_list: list[str] | None = None
    deny_list: list[str] | None = None
    page_min: int | None = None
    page_max: int | None = None


class ApplyReviewOptions(BaseOptions):
    # placeholder for future knobs
    pass


class RedactTabularOptions(BaseOptions):
    pii_method: str | None = "Local"
    columns: list[str] | None = None
    anon_strategy: str | None = "redact"
    allow_list: list[str] | None = None
    deny_list: list[str] | None = None
    language: str | None = "en"
    max_fuzzy_spelling_mistakes_num: int | None = 0
    do_initial_clean: bool | None = True
    llm_instruction: str | None = ""
    llm_entities: list[str] | None = None
    comprehend_entities: list[str] | None = None
    aws_access_key: str | None = ""
    aws_secret_key: str | None = ""


class SummariseOptions(BaseOptions):
    ocr_method: str | None = None
    summarisation_inference_method: str | None = None
    summarisation_format: Literal["concise", "detailed"] | None = None
    summarisation_context: str | None = None
    summarisation_additional_instructions: str | None = None
    summarisation_temperature: float | None = None
    summarisation_max_pages_per_group: int | None = None
    summarisation_api_key: str | None = None
    input_dir: str | None = None
    page_min: int | None = None
    page_max: int | None = None


class ArtifactFile(BaseModel):
    filename: str
    sha256: str
    size_bytes: int
    source: str | None = None


class ArtifactBundle(BaseModel):
    produced_by: str = Field(..., description="api_name used on the server")
    base_url: str
    files: list[ArtifactFile]
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

