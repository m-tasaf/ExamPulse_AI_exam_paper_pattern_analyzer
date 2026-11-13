# 🎓 ExamPulse AI — Past-Paper Pattern Analyzer & Study Assistant

**ExamPulse AI** is an AI-powered Streamlit dashboard that helps students analyze past-year exam papers, summarize every question, uncover recurring topic patterns across years, predict high-probability topics for the upcoming exam, and generate exam-ready model answers.

---

## 🎯 Problem & Who It Helps
Students preparing for university or competitive exams often spend hours manually reading through 3–5 years of past question papers trying to guess which chapters matter most. **ExamPulse AI** automates that workflow:
- **Summarizes** lengthy exam questions into clear, digestible points.
- **Quantifies** topic frequencies and year-over-year appearance patterns.
- **Predicts** high-yield topics for the next exam with data-backed reasoning.
- **Generates** structured model answers and key examiner bullet points on demand.

---

## 🧠 AI Implementation
- **Multimodal AI & Vision:** Directly accepts **PDFs, scanned documents, and images (PNG, JPEG, WEBP)**. Gemini's vision engine reads embedded diagrams, charts, circuit diagrams, and binary trees.
- **Model & SDK:** Uses the official `google-genai` Python SDK with **`gemini-3.5-flash-lite`** (with automatic failover) for ultra-fast, high-availability reasoning.
- **Structured JSON Output:** Uses `response_mime_type="application/json"` paired with **Pydantic** schemas (`PaperParseResult`, `PredictionReport`, `QuestionAnswer`) including diagram detection (`has_diagram`, `diagram_description`).
- **Hybrid Pipeline:** Combines **quantitative analysis (`pandas`)** for exact topic frequency counts and year-over-year heatmaps with **qualitative AI reasoning (Gemini)** for trend synthesis, next-exam predictions, and model answers.

---

## 🛠️ Tech Stack
| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core application logic |
| Dashboard UI | `streamlit` | Interactive web app, sidebar controls, tabs, metrics |
| AI / LLM | `google-genai` (`gemini-3.8-flash`) | Question parsing, topic normalization, predictions, answers |
| Schema Validation | `pydantic` | Enforces structured JSON output from Gemini |
| PDF Extraction | `pypdf` | Extracts text from uploaded past-year PDF papers |
| Data & Charts | `pandas` + `plotly` | Topic frequency tables, heatmaps, and interactive bar charts |
| Secrets | `python-dotenv` | Loads `GEMINI_API_KEY` safely from `.env` |

---

## 📁 Project Structure
```text
exam_question_summarizer_app/
├── app.py                # Main Streamlit dashboard UI
├── analyzer.py           # Gemini API calls & pandas pattern analysis
├── extractor.py          # PDF and text extraction helpers
├── requirements.txt      # Python dependencies
├── sample_papers/        # Built-in non-private sample papers (2022–2024)
│   ├── dsa_exam_2022.txt
│   ├── dsa_exam_2023.txt
│   └── dsa_exam_2024.txt
├── .env.example          # Template for your GEMINI_API_KEY
├── .gitignore            # Excludes .env and __pycache__/
└── README.md             # Project documentation
```

---

## 🚀 Setup & How to Run

### 1. Clone / Open the Project
```bash
cd exam_question_summarizer_app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Your Gemini API Key
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Add your key inside `.env`:
   ```env
   GEMINI_API_KEY=your_actual_key_here
   ```
   *(Alternatively, you can paste your API key directly into the password field in the Streamlit sidebar.)*

### 4. Launch the App
```bash
streamlit run app.py
```

### 5. Try It Immediately
1. In the sidebar, choose **Use Built-in Sample Papers** (3 sample Data Structures & Algorithms papers from 2022–2024) or upload your own `.pdf` / `.txt` past papers.
2. Click **🚀 Analyze Past Papers**.
3. Explore **Topic Patterns & Predictions**, **Question Summaries & Answers**, and **Export & Study Guide**.

---

## 🔒 Security Note
- Never commit `.env` or hardcode your `GEMINI_API_KEY` in source files.
- `.gitignore` is pre-configured to exclude `.env` and Python cache directories.

---

## 🙏 Credits & Acknowledgments
- [Streamlit](https://streamlit.io/) — Interactive Python web dashboard framework
- [Google Gemini API (`google-genai`)](https://ai.google.dev/) — LLM reasoning and structured output
- [pypdf](https://pypdf.readthedocs.io/) — Open-source PDF text extraction
- [pandas](https://pandas.pydata.org/) & [Plotly](https://plotly.com/python/) — Data analysis and interactive visualizations
