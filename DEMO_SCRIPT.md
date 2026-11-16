# 3–4 Minute Demo Video Script

A finished silent, captioned screencast is already included at
`demo/Resume_RAG_Demo.mp4` (3:16, 1920x1080). It walks through VS Code, the
ingestion code, live matcher runs, and the evaluation metrics. Use this script
only if you prefer to record a narrated version yourself.

This is a recording-ready walkthrough of the Resume RAG project. The demo can be
recorded with OBS Studio, Windows Game Bar (`Win + G`), or any screen recorder.
No camera is required.

## Before recording

1. Open the project in VS Code at `C:\Resume-RAG`.
2. Open a PowerShell terminal in the project root.
3. Increase the terminal font size so code and output are readable.
4. Close unrelated tabs and notifications.
5. Optional: run the pipeline once before recording so the embedding model is
   cached and the demo runs faster.

Useful files to have open in tabs:

- `src/resume_rag.py`
- `src/job_matcher.py`
- `notebooks/evaluation.ipynb`
- `outputs/job_002_machine_learning_engineer-results.json`

## Timed script

### 0:00–0:20 — Introduction

**On screen:** VS Code with the project tree visible.

**Say:**

> This is a Resume RAG system that retrieves the most relevant candidates for a
> job description and enforces mandatory hiring requirements. It uses HuggingFace
> embeddings, ChromaDB, semantic search, keyword matching, and an evaluation
> notebook. The dataset contains 30 synthetic resumes and 5 job descriptions.

### 0:20–1:05 — Architecture and project structure

**On screen:** Show the project tree.

**Say:**

> The pipeline has two main scripts. `resume_rag.py` handles ingestion: it loads
> PDFs, detects sections such as Summary, Skills, Experience, and Education,
> chunks each section, creates embeddings, extracts metadata, and stores
> everything in ChromaDB. `job_matcher.py` handles retrieval and ranking: it
> embeds the job description, retrieves candidates, combines semantic and keyword
> scores, and applies must-have requirements.

**Point out:**

- `resumes/` — 30 PDF resumes
- `job_descriptions/` — 5 job descriptions
- `resume_manifest.csv` — ground-truth metadata for evaluation
- `data/chroma_db/` — persistent vector database
- `notebooks/evaluation.ipynb` — metrics and analysis

### 1:05–1:50 — Document processing and embeddings

**On screen:** Open `src/resume_rag.py` and scroll to `split_sections`,
`create_document_chunks`, and `process_resume`.

**Say:**

> Resumes are split by section instead of arbitrary character counts. This keeps
> Education, Experience, and Skills coherent inside each chunk. Each chunk is
> embedded with the HuggingFace `all-MiniLM-L6-v2` model, and metadata such as
> candidate name, skills, experience years, and education is stored next to the
> embedding. The database currently contains 150 chunks from 30 resumes.

**Optional terminal command:**

```powershell
.\.venv\Scripts\python.exe .\src\resume_rag.py
```

**Expected output:** `Resumes processed: 30`, `Total chunks: 150`,
`Database records: 150`.

### 1:50–2:40 — Hybrid matching and mandatory requirements

**On screen:** Open `src/job_matcher.py` and show `score_candidate`,
`check_must_have`, and `contains_term`.

**Say:**

> The final score is a hybrid of semantic similarity at 60 percent, keyword
> overlap at 30 percent, and experience at 10 percent. Semantic similarity alone
> cannot award a mandatory skill. Every requirement group must pass, and groups
> use OR logic internally and AND logic across groups. If any mandatory group
> fails, the candidate score is capped at 59. The matcher also handles wording
> differences such as REST API versus REST APIs, but it does not accept generic
> words like deployed on their own.

### 2:40–3:20 — Live matching demo

**On screen:** Run the ML job, then the Python backend job.

```powershell
.\.venv\Scripts\python.exe .\src\job_matcher.py .\job_descriptions\job_002_machine_learning_engineer.txt
.\.venv\Scripts\python.exe .\demo\show_results.py .\outputs\job_002_machine_learning_engineer-results.json
```

**Say:**

> For the Machine Learning Engineer role, Sana Khan is qualified with a score of
> 79.91. Her resume explicitly shows deployed NLP classification models and
> production drift monitoring, along with Python, PyTorch, and TensorFlow.

```powershell
.\.venv\Scripts\python.exe .\src\job_matcher.py .\job_descriptions\job_001_python_backend_engineer.txt
.\.venv\Scripts\python.exe .\demo\show_results.py .\outputs\job_001_python_backend_engineer-results.json
```

**Say:**

> For the Python Backend role, Aarav Mehta's REST APIs are recognized, but he is
> still rejected because he has only 3 years of experience against a 5-year
> requirement and no production deployment evidence. This is intentional: the
> system does not inflate qualifications.

**Expected result:** 2 qualified candidates overall — Sana Khan and Kavya Menon.

### 3:20–3:45 — Evaluation and close

**On screen:** Open `notebooks/evaluation.ipynb` and show the metrics table and
charts.

**Say:**

> The evaluation notebook reports Precision@10, Recall@10, retrieval accuracy,
> qualification rate, and latency. Average Precision@10 is 0.44, average Recall@10
> is 0.733, and matching takes well under a second per job after model load. The
> notebook also runs six validation checks covering plural matching, deployment
> phrases, sourcing synonyms, strict stakeholder communication, networking
> evidence, and the exact qualified-candidate set.

**Close:**

> The project is fully reproducible: rebuild the database, run the matcher on any
> job description, and open the notebook for the final evaluation.

## One-command version

Instead of typing the commands individually:

```powershell
.\demo\run_demo.ps1
```

To rebuild the database first:

```powershell
.\demo\run_demo.ps1 -Rebuild
```

To demo a different job:

```powershell
.\demo\run_demo.ps1 -Job .\job_descriptions\job_004_talent_acquisition_specialist.txt
```

## Recording tips

- Keep the video between 3 and 4 minutes.
- Do not scroll too quickly; pause on the output tables.
- If model loading takes time, cut that section during editing.
- Show at least one accepted candidate and one rejected candidate.
- End on the notebook metrics and the two qualified candidates.
