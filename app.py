"""ExamPulse AI — Past-Paper Pattern Analyzer & Study Assistant (Streamlit Dashboard).

Now Multimodal:
- Reads and parses text, PDFs (including scanned papers), and images (JPEG, PNG, WEBP).
- Recognizes diagrams, graphs, charts, trees, and circuits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from analyzer import (
    CANDIDATE_MODELS,
    DEFAULT_MODEL,
    PredictionReport,
    compute_topic_stats,
    generate_model_answer,
    generate_predictions,
    get_gemini_client,
    parse_questions_from_file,
    parse_questions_from_text,
)
from extractor import (
    extract_text_from_pdf,
    get_file_mime_type,
    infer_year_from_filename,
    is_image_file,
    is_pdf_file,
)

load_dotenv()

SAMPLE_DIR = Path(__file__).parent / "sample_papers"

st.set_page_config(
    page_title="ExamPulse AI — Exam Pattern Analyzer",
    page_icon="🎓",
    layout="wide",
)


def _init_session_state() -> None:
    defaults = {
        "questions_df": pd.DataFrame(),
        "topic_stats_df": pd.DataFrame(),
        "year_topic_matrix": pd.DataFrame(),
        "prediction_report": None,
        "answers_cache": {},
        "analyzed_subject": "",
        "use_samples": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_sample_papers() -> list[dict[str, Any]]:
    """Return list of paper dicts from sample_papers/."""
    papers: list[dict[str, Any]] = []
    if not SAMPLE_DIR.exists():
        return papers
    for path in sorted(SAMPLE_DIR.glob("*.txt")):
        year = infer_year_from_filename(path.name, default_year=2024)
        raw_bytes = path.read_bytes()
        papers.append(
            {
                "year": year,
                "filename": path.name,
                "bytes": raw_bytes,
                "mime_type": "text/plain",
                "extracted_text": raw_bytes.decode("utf-8", errors="replace"),
            }
        )
    return papers


def _run_analysis(
    papers: list[dict[str, Any]],
    subject: str,
    api_key: str,
    model_name: str,
) -> None:
    """Execute the multimodal AI & pandas pipeline and store results in session_state."""
    client = get_gemini_client(api_key)
    all_rows: list[dict] = []

    progress = st.progress(0, text="Analyzing exam papers with Multimodal Gemini...")
    total_steps = len(papers) + 1

    for idx, paper in enumerate(papers, start=1):
        name = paper["filename"]
        year = paper["year"]
        progress.progress(
            idx / total_steps,
            text=f"Inspecting diagrams & extracting questions from {name} ({year})...",
        )
        rows = parse_questions_from_file(
            client=client,
            file_bytes=paper["bytes"],
            mime_type=paper["mime_type"],
            filename=name,
            year=year,
            subject=subject,
            model_name=model_name,
            extracted_text=paper.get("extracted_text"),
        )
        all_rows.extend(rows)

    if not all_rows:
        progress.empty()
        st.warning("No questions could be extracted from the provided files.")
        return

    questions_df = pd.DataFrame(all_rows).sort_values(["year", "question_number"]).reset_index(drop=True)
    topic_stats_df, year_topic_matrix = compute_topic_stats(questions_df)

    progress.progress(
        (total_steps - 0.5) / total_steps,
        text="Analyzing topic patterns and generating next-exam predictions...",
    )
    prediction_report = generate_predictions(
        client, topic_stats_df, questions_df, subject=subject, model_name=model_name
    )
    progress.empty()

    st.session_state["questions_df"] = questions_df
    st.session_state["topic_stats_df"] = topic_stats_df
    st.session_state["year_topic_matrix"] = year_topic_matrix
    st.session_state["prediction_report"] = prediction_report
    st.session_state["answers_cache"] = {}
    st.session_state["analyzed_subject"] = subject


def _build_markdown_report(
    subject: str,
    topic_stats_df: pd.DataFrame,
    report: PredictionReport,
) -> str:
    """Format analysis and predictions into a downloadable Markdown study guide."""
    cols = ["topic", "total_questions", "years_appeared", "share_pct", "pattern"]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    table_rows = [header, separator]
    for _, r in topic_stats_df[cols].iterrows():
        table_rows.append(
            f"| {r['topic']} | {r['total_questions']} | {r['years_appeared']} | {r['share_pct']}% | {r['pattern']} |"
        )
    topic_table_md = "\n".join(table_rows)

    lines = [
        f"# ExamPulse AI Study Report — {subject}",
        "",
        "## Executive Trend Summary",
        report.overall_trend_summary,
        "",
        "## Topic Frequency Summary",
        topic_table_md,
        "",
        "## Next Exam Predictions & Model Answers",
    ]
    for i, pred in enumerate(report.predictions, start=1):
        lines.extend(
            [
                f"### {i}. {pred.topic} ({pred.probability} Probability)",
                f"**Why it's likely:** {pred.reasoning}",
                "",
                f"**Practice Question:** {pred.sample_predicted_question}",
                "",
                f"**Model Answer:**\n{pred.model_answer}",
                "",
                "---",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    _init_session_state()

    st.title("🎓 ExamPulse AI — Past-Paper Pattern Analyzer")
    st.caption(
        "Upload past-year exam papers (PDFs, Images, or Text) to summarize questions, "
        "recognize diagrams and charts, uncover recurring patterns, and predict next-exam topics."
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        subject = st.text_input(
            "Exam / Subject Name",
            value="Data Structures & Algorithms",
            help="Helps Gemini normalize topics accurately for your course.",
        )

        # Check if key is available via .env or Streamlit Cloud Secrets (st.secrets)
        has_secret_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
        try:
            if "GEMINI_API_KEY" in st.secrets and str(st.secrets["GEMINI_API_KEY"]).strip():
                has_secret_key = True
        except Exception:
            pass

        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Loaded from Secrets/.env" if has_secret_key else "Paste your GEMINI_API_KEY",
            help="Stored only in your current session. Falls back to Streamlit Secrets or .env.",
        )

        selected_model = st.selectbox(
            "Gemini Model",
            options=CANDIDATE_MODELS,
            index=0,
            help="gemini-3.5-flash-lite runs in ~2 seconds with high quota availability and full vision support.",
        )

        st.divider()
        st.header("📄 Input Past Papers")
        input_mode = st.radio(
            "Choose input method",
            options=[
                "Upload Files (PDF / Images / TXT)",
                "Paste Question Text",
                "Use Built-in Sample Papers",
            ],
        )

        papers_to_analyze: list[dict[str, Any]] = []

        if input_mode == "Upload Files (PDF / Images / TXT)":
            uploaded_files = st.file_uploader(
                "Upload past-year exam files",
                type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
                accept_multiple_files=True,
                help="Accepts PDFs, scanned documents, and images (diagrams, photos of papers).",
            )
            if uploaded_files:
                st.markdown("**Assign Exam Year for Each File:**")
                for idx, uf in enumerate(uploaded_files):
                    default_yr = infer_year_from_filename(uf.name, default_year=2024 - idx)
                    yr = st.number_input(
                        f"Year for `{uf.name}`",
                        min_value=2000,
                        max_value=2035,
                        value=default_yr,
                        step=1,
                        key=f"year_{uf.name}_{idx}",
                    )
                    raw_bytes = uf.getvalue()
                    mime = get_file_mime_type(uf.name)
                    extracted_text = ""
                    if is_pdf_file(uf.name):
                        extracted_text = extract_text_from_pdf(raw_bytes)
                    elif mime == "text/plain":
                        extracted_text = raw_bytes.decode("utf-8", errors="replace")

                    papers_to_analyze.append(
                        {
                            "year": int(yr),
                            "filename": uf.name,
                            "bytes": raw_bytes,
                            "mime_type": mime,
                            "extracted_text": extracted_text,
                        }
                    )

        elif input_mode == "Paste Question Text":
            paste_year = st.number_input("Exam Year", min_value=2000, max_value=2035, value=2024, step=1)
            pasted_text = st.text_area(
                "Paste exam questions here",
                height=200,
                placeholder="Q1. Explain Dynamic Programming...\nQ2. Trace Dijkstra's algorithm...",
            )
            if pasted_text.strip():
                papers_to_analyze.append(
                    {
                        "year": int(paste_year),
                        "filename": f"Pasted_{paste_year}.txt",
                        "bytes": pasted_text.strip().encode("utf-8"),
                        "mime_type": "text/plain",
                        "extracted_text": pasted_text.strip(),
                    }
                )

        else:
            sample_papers = _load_sample_papers()
            if sample_papers:
                st.info(
                    f"Loaded {len(sample_papers)} sample DSA papers: "
                    + ", ".join(str(p["year"]) for p in sample_papers)
                )
                papers_to_analyze = sample_papers
            else:
                st.warning("No sample papers found in `sample_papers/`.")

        analyze_clicked = st.button(
            "Analyze Past Papers",
            type="primary",
            use_container_width=True,
            disabled=len(papers_to_analyze) == 0,
        )

    if analyze_clicked:
        try:
            _run_analysis(
                papers_to_analyze,
                subject=subject,
                api_key=api_key_input,
                model_name=selected_model,
            )
            st.success("Analysis complete!")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    # --- Main Dashboard ---
    questions_df: pd.DataFrame = st.session_state["questions_df"]
    topic_stats_df: pd.DataFrame = st.session_state["topic_stats_df"]
    year_topic_matrix: pd.DataFrame = st.session_state["year_topic_matrix"]
    report: PredictionReport | None = st.session_state["prediction_report"]

    if questions_df.empty or topic_stats_df.empty or report is None:
        st.info(
            "👈 Select **Use Built-in Sample Papers** (or upload your PDF/Image/TXT files) in the sidebar "
            "and click **Analyze Past Papers** to get started."
        )
        return

    # KPI Row
    top_topic = topic_stats_df.iloc[0]["topic"]
    top_count = int(topic_stats_df.iloc[0]["total_questions"])
    years_covered = sorted(questions_df["year"].unique().tolist())
    diagram_count = (
        int(questions_df["has_diagram"].sum())
        if "has_diagram" in questions_df.columns
        else 0
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Questions Analyzed", len(questions_df))
    k2.metric("Years Covered", f"{len(years_covered)} ({min(years_covered)}–{max(years_covered)})")
    k3.metric("#1 Most Frequent Topic", top_topic, f"{top_count} questions")
    k4.metric("Diagram / Chart Questions", f"{diagram_count} 🖼️")

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Topic Patterns & Predictions",
            "📝 Question Summaries & Answers",
            "⬇️ Export & Study Guide",
        ]
    )

    # --- Tab 1: Topic Patterns & Predictions ---
    with tab1:
        st.subheader("Topic Frequency & Year-over-Year Patterns")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_bar = px.bar(
                topic_stats_df.sort_values("total_questions", ascending=True),
                x="total_questions",
                y="topic",
                color="pattern",
                orientation="h",
                title="Questions per Topic (Colored by Pattern)",
                labels={"total_questions": "Question Count", "topic": "Topic", "pattern": "Pattern"},
                text="total_questions",
            )
            fig_bar.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_chart2:
            if not year_topic_matrix.empty:
                fig_heat = px.imshow(
                    year_topic_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="Topic Appearance Heatmap Across Years",
                    labels={"x": "Exam Year", "y": "Topic", "color": "Questions"},
                )
                fig_heat.update_xaxes(type="category")
                fig_heat.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("📋 View Full Topic Frequency Table", expanded=False):
            st.dataframe(
                topic_stats_df[
                    ["topic", "total_questions", "years_appeared", "share_pct", "year_coverage_pct", "years_list", "pattern"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.subheader("🔮 Next Exam Predictions (AI-Powered)")
        st.info(f"**Overall Exam Trend:** {report.overall_trend_summary}")

        badge_map = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟡 Low"}
        for pred in report.predictions:
            badge = badge_map.get(pred.probability, pred.probability)
            with st.container(border=True):
                c_left, c_right = st.columns([4, 1])
                c_left.markdown(f"### {pred.topic}")
                c_right.markdown(f"**Probability:** {badge}")
                st.markdown(f"**Why it's predicted:** {pred.reasoning}")
                st.markdown(f"**Likely Practice Question:** {pred.sample_predicted_question}")
                with st.expander("View Model Answer"):
                    st.markdown(pred.model_answer)

    # --- Tab 2: Question Summaries & Answers ---
    with tab2:
        st.subheader("Browse Past Questions & Generate Model Answers")
        f1, f2, f3 = st.columns(3)
        selected_years = f1.multiselect(
            "Filter by Year",
            options=years_covered,
            default=years_covered,
        )
        all_topics = sorted(questions_df["topic"].unique().tolist())
        selected_topics = f2.multiselect(
            "Filter by Topic",
            options=all_topics,
            default=all_topics,
        )
        all_types = sorted(questions_df["question_type"].unique().tolist())
        selected_types = f3.multiselect(
            "Filter by Question Type",
            options=all_types,
            default=all_types,
        )

        filtered_df = questions_df[
            questions_df["year"].isin(selected_years)
            & questions_df["topic"].isin(selected_topics)
            & questions_df["question_type"].isin(selected_types)
        ]

        st.caption(f"Showing {len(filtered_df)} of {len(questions_df)} questions")

        for idx, row in filtered_df.iterrows():
            marks_str = f" | {row['marks']}" if pd.notna(row["marks"]) and row["marks"] else ""
            diagram_badge = " 🖼️ [Has Diagram/Graph]" if row.get("has_diagram") else ""
            header = (
                f"[{row['year']}] {row['question_number']} — {row['topic']} "
                f"({row['question_type']}{marks_str}){diagram_badge}"
            )
            with st.expander(header):
                st.markdown(f"**Question Summary:** {row['question_summary']}")
                if row.get("has_diagram") and pd.notna(row.get("diagram_description")):
                    st.info(f"📊 **Diagram / Visual Content:** {row['diagram_description']}")

                cache_key = f"{row['year']}_{row['question_number']}_{idx}"
                cached_answer = st.session_state["answers_cache"].get(cache_key)

                if cached_answer is None:
                    if st.button("✨ Generate Model Answer", key=f"btn_{cache_key}"):
                        try:
                            with st.spinner("Generating exam-ready answer with Gemini..."):
                                client = get_gemini_client(api_key_input)
                                ans = generate_model_answer(
                                    client=client,
                                    question_summary=row["question_summary"],
                                    topic=row["topic"],
                                    question_type=row["question_type"],
                                    subject=st.session_state["analyzed_subject"],
                                    model_name=selected_model,
                                )
                                st.session_state["answers_cache"][cache_key] = ans
                                st.rerun()
                        except Exception as exc:
                            st.error(f"Could not generate answer: {exc}")
                else:
                    st.markdown("**Model Answer:**")
                    st.markdown(cached_answer.model_answer)
                    if cached_answer.key_points:
                        st.markdown("**Key Examiner Points:**")
                        for pt in cached_answer.key_points:
                            st.markdown(f"- {pt}")

    # --- Tab 3: Export & Study Guide ---
    with tab3:
        st.subheader("Download Your Analysis")
        csv_data = questions_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Analyzed Questions (CSV)",
            data=csv_data,
            file_name="exampulse_questions.csv",
            mime="text/csv",
        )

        md_report = _build_markdown_report(
            subject=st.session_state["analyzed_subject"],
            topic_stats_df=topic_stats_df,
            report=report,
        )
        st.download_button(
            label="📥 Download Study & Prediction Report (Markdown)",
            data=md_report.encode("utf-8"),
            file_name="exampulse_study_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
