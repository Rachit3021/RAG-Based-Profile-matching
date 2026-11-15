# Resume RAG - Synthetic Resume Matching Project

A retrieval-augmented resume screening project built on 30 synthetic PDF resumes
and 5 job descriptions. It retrieves the most relevant candidates with
sentence-transformer embeddings and ChromaDB, then enforces mandatory
requirements with explicit resume evidence.

All names, emails, phone numbers, organizations, and career histories are
synthetic. Do not use this dataset or its outputs for real hiring decisions.

## What the project does

1. Parses each PDF, splits it into sections and chunks, embeds the chunks with
   `sentence-transformers/all-MiniLM-L6-v2`, and stores them in ChromaDB.
2. Retrieves the full resume pool for each job description and ranks a top-10
   list using semantic similarity (60%), keyword overlap (30%), and experience
   (10%).
3. Applies mandatory requirements with OR logic inside a requirement group and
   AND logic across groups.
4. Caps candidates that fail any mandatory requirement at a score of 59.
5. Writes per-job JSON reports plus evaluation CSV reports.

## Project structure

```
resumes/                      30 synthetic PDF resumes
job_descriptions/             5 job descriptions (.txt)
resume_manifest.csv           Ground-truth metadata and category labels
src/resume_rag.py             PDF ingestion and ChromaDB build
src/job_matcher.py            Retrieval, scoring, and requirement matching
notebooks/evaluation.ipynb    Executed final evaluation and validation checks
notebooks/archive/            Original notebook retained for reference
demo/show_results.py          Concise candidate summary for demos
demo/run_demo.ps1             One-command demo runner
DEMO_SCRIPT.md                3-4 minute recording script and narration
SUBMISSION_CHECKLIST.md       Assignment requirement-to-file mapping
outputs/                      JSON and CSV reports
data/chroma_db/               Persistent ChromaDB store
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the pipeline

From the project root:

```powershell
# 1. Build or refresh the vector database from the PDF resumes
.\.venv\Scripts\python.exe .\src\resume_rag.py

# 2. Match every job description. Each run writes its own report.
Get-ChildItem .\job_descriptions\job_*.txt | ForEach-Object {
    .\.venv\Scripts\python.exe .\src\job_matcher.py $_.FullName
}

# 3. Re-execute the evaluation notebook
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace .\notebooks\evaluation.ipynb
```

A single job can also be run directly:

```powershell
.\.venv\Scripts\python.exe .\src\job_matcher.py .\job_descriptions\job_001_python_backend_engineer.txt
```

Each matcher run writes `outputs/<job_name>-results.json` and refreshes
`outputs/matching_results.json` with a copy of the latest run.

## Current evaluation summary

The pipeline retrieves 30 resumes per job and evaluates the top 10.

| Job | Qualified in top 10 | Qualified candidate(s) |
| --- | ---: | --- |
| Python Backend Engineer | 0 | - |
| Machine Learning Engineer | 1 | Sana Khan |
| Production Engineer | 1 | Kavya Menon |
| Talent Acquisition Specialist | 0 | - |
| Cybersecurity Analyst | 0 | - |
| **Total** | **2** | |

Average retrieval metrics from the latest run: **Precision@10 = 0.44**,
**Recall@10 = 0.733**, **retrieval accuracy = 44.0%** (category match rate in the
top 10). Average matching latency is about **0.30 seconds** per job after the
embedding model is loaded.

Key near-misses that remain rejected:

- **Aarav Mehta (Python Backend, 59.0):** `Built REST APIs` now matches the REST
  requirement, but he lacks production deployment evidence and has 3 years of
  experience against a 5-year requirement.
- **Diya Sharma (Python Backend, 59.0):** Has 5 years, Django/Flask, PostgreSQL,
  and Docker, but no REST API or deployment evidence.
- **Riya Kapoor (Talent Acquisition, 59.0):** `Sourcing` now matches candidate
  sourcing, but stakeholder communication has no resume evidence.
- **Pranav Kulkarni (Cybersecurity, 59.0):** Matches five of six groups but has
  no networking evidence.

## Matching rules

- Requirement groups use OR within a group and AND across groups.
- Plural handling applies only to the final word of a phrase, so `REST API`
  matches `REST APIs` without matching unrelated words.
- The ML deployment group accepts specific phrases such as
  `deployed NLP classification models`, `deployed ML models`, and
  `monitored production drift`. Bare `deployed` is deliberately rejected.
- Candidate sourcing accepts `sourcing`, `candidate sourcing`,
  `sourced candidates`, and related phrases.
- Stakeholder communication remains a strict requirement.
- Semantic similarity cannot award a missing mandatory requirement.

## Outputs

| File | Contents |
| --- | --- |
| `outputs/job_00X_*-results.json` | Per-job ranked candidates, scores, evidence, and missing requirements |
| `outputs/matching_results.json` | Copy of the most recent job run |
| `outputs/evaluation_metrics.csv` | Retrieval and qualification metrics per job |
| `outputs/final_evaluation_report.csv` | Full evaluation table |
| `outputs/qualified_candidates.csv` | Candidates passing every mandatory requirement |
| `outputs/missing_requirements.csv` | Top-10 candidates and their missing requirements |

## Evaluation notebook

`notebooks/evaluation.ipynb` is fully executed and contains:

- retrieval quality and qualification tables
- charts for Precision@10, Recall@10, retrieval accuracy, qualified counts, and latency
- top-five candidates per job with missing requirements
- six automated validation checks covering plural matching, deployment phrase
  matching, sourcing synonyms, strict stakeholder communication, networking
  evidence, and the exact qualified-candidate set
- saved CSV reports

## Demo video

The 3-4 minute demo script is in `DEMO_SCRIPT.md`. It includes timestamps,
narration, exact commands, and expected output. Record it with OBS Studio,
Windows Game Bar (`Win + G`), or any screen recorder.

A one-command demo is also available:

```powershell
# Demo the ML job using the existing database
.\demo\run_demo.ps1

# Rebuild the database first
.\demo\run_demo.ps1 -Rebuild

# Demo a different job
.\demo\run_demo.ps1 -Job .\job_descriptions\job_004_talent_acquisition_specialist.txt
```

`demo/show_results.py` prints a concise table of the top five candidates with
scores, experience, matched evidence, missing requirements, and an excerpt.

## Assignment mapping

`SUBMISSION_CHECKLIST.md` maps every Part A and Part B requirement to the
implementation that satisfies it, including the required output fields and all
deliverables.

## Submission package

A clean package is available at `Resume-RAG-Submission.zip` containing the code,
dataset, executed notebook, outputs, documentation, demo scripts, and the built
ChromaDB store. It excludes `.venv`, `__pycache__`, and notebook checkpoints.

## Limitations

- The dataset is synthetic and small (30 resumes, 5 jobs).
- Matching is keyword- and phrase-based, so unseen synonyms require explicit
  alternatives.
- Resume-category labels are a coarse proxy for retrieval relevance.
- A production system would add recency weighting, skill-level extraction,
  score calibration, and human review.
