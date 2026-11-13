"""Gemini API integration and pandas-based pattern analysis for ExamPulse AI."""

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

# Primary and fallback models verified available and fast
CANDIDATE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]
DEFAULT_MODEL = CANDIDATE_MODELS[0]


class ParsedQuestion(BaseModel):
    """Structured representation of a single exam question."""

    question_number: str = Field(description="Question label, e.g., Q1, Q2(a), 3.")
    question_summary: str = Field(
        description="1-2 sentence concise summary of what the question is asking."
    )
    topic: str = Field(
        description="Normalized, concise topic or chapter name (Title Case, 2-4 words)."
    )
    question_type: str = Field(
        description="Category such as Theory, Numerical, Coding, Case Study, or Short Answer."
    )
    marks: Optional[str] = Field(
        default=None, description="Marks or points if mentioned, otherwise null."
    )
    has_diagram: bool = Field(
        default=False,
        description="True if the question includes or explicitly references a diagram, graph, chart, circuit, tree, or figure.",
    )
    diagram_description: Optional[str] = Field(
        default=None,
        description="Concise description of what is depicted in the diagram/chart/figure, or null if none.",
    )


class PaperParseResult(BaseModel):
    """Structured list of questions extracted from one exam paper."""

    questions: list[ParsedQuestion]


class TopicPrediction(BaseModel):
    """AI prediction for a high-yield topic in the upcoming exam."""

    topic: str = Field(description="Topic name matching the analyzed papers.")
    probability: str = Field(description="High, Medium, or Low.")
    reasoning: str = Field(
        description="Data-backed explanation citing frequency, recency, or alternating-year pattern."
    )
    sample_predicted_question: str = Field(
        description="A realistic practice question likely to appear on the next exam."
    )
    model_answer: str = Field(
        description="Clear, well-structured, exam-ready model answer for the predicted question."
    )


class PredictionReport(BaseModel):
    """Overall trend analysis and next-exam predictions."""

    overall_trend_summary: str = Field(
        description="2-3 sentence executive summary of how the exam has evolved across years."
    )
    predictions: list[TopicPrediction] = Field(
        description="Top 4-6 predicted topics for the next exam, ordered by likelihood."
    )


class QuestionAnswer(BaseModel):
    """Detailed model answer for a specific past-year question."""

    model_answer: str = Field(
        description="Comprehensive, exam-ready answer with clear steps or structure."
    )
    key_points: list[str] = Field(
        description="3-5 bullet points highlighting the key concepts examiners look for."
    )


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Create a Google GenAI client using user input, st.secrets, or .env."""
    key_from_secrets = ""
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            key_from_secrets = str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    resolved_key = (
        (api_key or "").strip()
        or key_from_secrets
        or os.getenv("GEMINI_API_KEY", "").strip()
    )
    if not resolved_key:
        raise ValueError(
            "Gemini API key is missing. Enter it in the sidebar or set GEMINI_API_KEY in .env / Streamlit Secrets."
        )
    return genai.Client(api_key=resolved_key)


def _generate_structured(
    client: genai.Client,
    contents: str | list[types.Part | str],
    schema: type[BaseModel],
    temperature: float = 0.2,
    selected_model: Optional[str] = None,
) -> BaseModel:
    """Call Gemini with structured JSON output, automatic retry, and fallback models on 503/404."""
    models_to_try = [selected_model] if selected_model else CANDIDATE_MODELS
    # If selected_model specified, add fallback models after it
    if selected_model:
        models_to_try += [m for m in CANDIDATE_MODELS if m != selected_model]

    # Normalize contents into a list of Parts or str
    content_payload = contents if isinstance(contents, list) else [contents]

    last_error: Exception | None = None

    for model_name in models_to_try:
        # Try up to 2 attempts per model with exponential backoff on 503
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=content_payload,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=temperature,
                    ),
                )
                if getattr(response, "parsed", None) is not None:
                    return response.parsed
                if response.text:
                    return schema.model_validate_json(response.text)
                raise RuntimeError(f"Model {model_name} returned an empty response.")
            except errors.APIError as exc:
                last_error = exc
                # 503 Service Unavailable / high demand -> brief wait and retry or switch model
                if exc.code in (503, 429) and attempt == 0:
                    time.sleep(2)
                    continue
                # If 404 or 503 persists, break inner loop to try next fallback model
                break
            except Exception as exc:
                last_error = exc
                break

    raise RuntimeError(
        f"All candidate models failed. Last error: {last_error}"
    ) from last_error


def parse_questions_from_file(
    client: genai.Client,
    file_bytes: bytes,
    mime_type: str,
    filename: str,
    year: int,
    subject: str,
    model_name: Optional[str] = None,
    extracted_text: Optional[str] = None,
) -> list[dict]:
    """Extract and classify all questions from an exam paper (PDF, Image, or Text) using Multimodal Gemini.

    Capable of recognizing diagrams, graphs, charts, circuit diagrams, trees, and handwritten/scanned questions.
    """
    prompt_text = f"""
You are an expert academic exam analyst for the subject: "{subject}".
Analyze this {year} exam question paper carefully and extract EVERY distinct question or sub-question.

Visual & Diagram Understanding Instructions:
- Carefully inspect any diagrams, graphs, flowcharts, finite automata, binary trees, tables, or circuits shown on the page/image.
- If a question refers to an image, diagram, circuit, or graph, set `has_diagram=true` and summarize what the diagram depicts in `diagram_description` (e.g., "Directed weighted graph with 5 vertices", "Binary search tree with root 15").
- Transcribe the core question and any numerical values or labels present in the diagram.
- Summarize each question clearly in 1-2 sentences (`question_summary`).
- Assign a clean, normalized `topic` name in Title Case (e.g., "Dynamic Programming", "Graph Algorithms", "Database Normalization") so identical concepts across different years share the exact same topic label.
- Classify `question_type` into one of: Theory, Numerical, Coding, Case Study, Short Answer.
- Extract `marks` if visible (e.g., "10 marks"), otherwise null.
"""

    contents: list[types.Part | str] = []

    if mime_type.startswith("image/"):
        # For PNG, JPEG, WEBP: send image bytes directly for visual inspection
        contents.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
        contents.append(prompt_text)
    elif mime_type == "application/pdf":
        # Check if we have extracted digital text or if it's scanned / diagram-heavy
        if extracted_text and len(extracted_text.strip()) > 100:
            # We have text; also pass PDF bytes so Gemini can see diagrams
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"))
            contents.append(
                f"{prompt_text}\n\nExtracted Text Reference:\n\"\"\"\n{extracted_text[:30000]}\n\"\"\""
            )
        else:
            # Scanned or image-only PDF: pass PDF directly to Gemini Vision
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"))
            contents.append(prompt_text)
    else:
        # Plain text
        text_content = extracted_text or file_bytes.decode("utf-8", errors="replace")
        contents.append(f"{prompt_text}\n\nExam Paper Text ({year}):\n\"\"\"\n{text_content[:40000]}\n\"\"\"")

    result: PaperParseResult = _generate_structured(  # type: ignore[assignment]
        client, contents, PaperParseResult, temperature=0.1, selected_model=model_name
    )

    rows: list[dict] = []
    for q in result.questions:
        rows.append(
            {
                "year": year,
                "question_number": q.question_number,
                "topic": q.topic.strip(),
                "question_type": q.question_type.strip(),
                "marks": q.marks,
                "has_diagram": q.has_diagram,
                "diagram_description": q.diagram_description,
                "question_summary": q.question_summary.strip(),
            }
        )
    return rows


def parse_questions_from_text(
    client: genai.Client,
    paper_text: str,
    year: int,
    subject: str,
    model_name: Optional[str] = None,
) -> list[dict]:
    """Convenience wrapper for backward compatibility with plain text inputs."""
    return parse_questions_from_file(
        client=client,
        file_bytes=paper_text.encode("utf-8"),
        mime_type="text/plain",
        filename=f"pasted_{year}.txt",
        year=year,
        subject=subject,
        model_name=model_name,
        extracted_text=paper_text,
    )


def compute_topic_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute quantitative topic frequencies and year-by-topic matrix using pandas."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    total_questions = len(df)
    all_years = sorted(df["year"].unique().tolist())
    num_years = len(all_years)
    latest_exam_year = max(all_years)

    grouped = (
        df.groupby("topic", as_index=False)
        .agg(
            total_questions=("question_summary", "count"),
            years_appeared=("year", "nunique"),
            latest_year=("year", "max"),
            years_list=("year", lambda s: ", ".join(str(y) for y in sorted(set(s)))),
        )
        .sort_values(by=["total_questions", "years_appeared"], ascending=False)
        .reset_index(drop=True)
    )

    grouped["share_pct"] = (grouped["total_questions"] / total_questions * 100).round(1)
    grouped["year_coverage_pct"] = (grouped["years_appeared"] / num_years * 100).round(1)

    def _classify_pattern(row: pd.Series) -> str:
        if num_years >= 2 and row["years_appeared"] == num_years:
            return "Every Year"
        if row["years_appeared"] >= 2:
            return "Recurring"
        if row["latest_year"] == latest_exam_year:
            return "Recent Focus"
        return "Occasional"

    grouped["pattern"] = grouped.apply(_classify_pattern, axis=1)

    year_topic_matrix = (
        df.pivot_table(
            index="topic",
            columns="year",
            values="question_summary",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(grouped["topic"])
    )

    return grouped, year_topic_matrix


def generate_predictions(
    client: genai.Client,
    topic_stats_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    subject: str,
    model_name: Optional[str] = None,
) -> PredictionReport:
    """Use Gemini to combine frequency data and past questions into next-exam predictions."""
    stats_csv = topic_stats_df.to_csv(index=False)
    sample_questions = questions_df[
        [
            col
            for col in [
                "year",
                "topic",
                "question_type",
                "has_diagram",
                "diagram_description",
                "question_summary",
            ]
            if col in questions_df.columns
        ]
    ].to_string(index=False)

    prompt = f"""
You are an expert exam strategist for "{subject}".
Below is the quantitative topic frequency table across past exam years, followed by the summarized past questions.

Topic Frequency Table:
{stats_csv}

Past Questions:
{sample_questions[:25000]}

Task:
1. Provide a concise `overall_trend_summary` explaining the exam's patterns (e.g., consistent core topics, shifting emphasis, theory vs. practical balance).
2. Predict the top 5 topics most likely to appear in the NEXT exam (`predictions`), ordered from highest to lowest probability.
   - Base your `reasoning` directly on the frequency counts, year-over-year consistency, or recency gaps (e.g., alternating years).
   - Write a realistic `sample_predicted_question` for each topic.
   - Provide a clear, exam-ready `model_answer` for each predicted question.
"""
    return _generate_structured(  # type: ignore[return-value]
        client, prompt, PredictionReport, temperature=0.3, selected_model=model_name
    )


def generate_model_answer(
    client: genai.Client,
    question_summary: str,
    topic: str,
    question_type: str,
    subject: str,
    model_name: Optional[str] = None,
) -> QuestionAnswer:
    """Generate a structured, exam-ready model answer for a specific past question."""
    prompt = f"""
You are a subject-matter expert and university examiner for "{subject}".
Write a clear, accurate, exam-ready model answer for the following past exam question.

Topic: {topic}
Question Type: {question_type}
Question: {question_summary}

Include:
- `model_answer`: A well-structured answer (with steps, formulas, or examples where appropriate).
- `key_points`: 3-5 concise bullet points summarizing the core concepts required for full marks.
"""
    return _generate_structured(  # type: ignore[return-value]
        client, prompt, QuestionAnswer, temperature=0.2, selected_model=model_name
    )
