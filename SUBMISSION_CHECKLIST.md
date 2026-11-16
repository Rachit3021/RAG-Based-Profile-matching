# Submission Checklist

Mapping of the assignment brief to the implementation in this repository.

## Dataset

| Requirement | Status | Evidence |
| --- | --- | --- |
| 30+ diverse resumes | Complete | `resumes/` contains 30 synthetic PDFs across 5 categories |
| 5+ job descriptions | Complete | `job_descriptions/` contains 5 text files |
| Ground-truth metadata | Complete | `resume_manifest.csv` (name, role, experience, education, skills, category) |

## Part A — RAG System Setup (50%)

| Requirement | Status | Implementation |
| --- | --- | --- |
| Create `resume_rag.py` | Complete | `src/resume_rag.py` |
| Load resumes using file-system tools | Complete | `main()` reads `resumes/*.pdf` with `Path.glob` |
| Chunk documents intelligently | Complete | `split_sections()` and `create_document_chunks()` preserve Summary, Skills, Experience, Education, Projects, Certifications |
| Generate embeddings | Complete | `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` |
| Store in a vector database | Complete | ChromaDB collection `resume_collection` at `data/chroma_db/` |
| Extract Name | Complete | `extract_metadata()` uses manifest, with spaCy PERSON fallback |
| Extract Skills | Complete | Manifest skills, with keyword fallback in `extract_skills()` |
| Extract Experience Years | Complete | Manifest value, with regex fallback in `extract_experience()` |
| Extract Education | Complete | Manifest value, with regex fallback in `extract_education()` |
| Store metadata alongside embeddings | Complete | `metadatas` passed to `collection.add()` with name, path, section, skills, experience, education, chunk index |

## Part B — Job Matching Engine (50%)

| Requirement | Status | Implementation |
| --- | --- | --- |
| Create `job_matcher.py` | Complete | `src/job_matcher.py` |
| Accept a job description as input | Complete | CLI argument: `python src/job_matcher.py <job.txt>` |
| Convert JD to embedding | Complete | `model.encode([jd])` in `retrieve_candidates()` |
| Retrieve top-K similar resumes (K=10) | Complete | `retrieve_candidates()` groups all chunks by resume; `match_resumes()` ranks and returns `top_matches[:10]` |
| Hybrid search: semantic + keyword | Complete | `score_candidate()` combines 60% semantic, 30% keyword, 10% experience |
| Score matches on 0–100 scale | Complete | `match_score` clamped to 0–100 |
| Match reasoning: which sections matched | Complete | `reasoning`, `matched_sections`, `matched_skills`, `relevant_excerpts` |
| Filter by must-have requirements | Complete | `check_must_have()` applies AND across groups; `qualified_matches` contains only passing candidates |
| Enforce experience thresholds | Complete | `extract_required_experience()` and `check_must_have()` |
| Prevent false positives on wording variants | Complete | `contains_term()` handles plurals such as `REST API` / `REST APIs` with word-boundary guards |

## Required output format

Each `outputs/job_*.json` contains the required shape, with additional diagnostic
fields:

| Required field | Present | Source |
| --- | --- | --- |
| `job_description` | Yes | Job description filename |
| `top_matches` | Yes | Top 10 ranked candidates |
| `candidate_name` | Yes | Resume metadata |
| `resume_path` | Yes | Resume metadata |
| `match_score` | Yes | Hybrid score, 0–100 |
| `matched_skills` | Yes | Keyword overlap |
| `relevant_excerpts` | Yes | Retrieved resume chunks |
| `reasoning` | Yes | Semantic, keyword, experience, and mandatory-check summary |

Additional useful fields: `qualified_matches`, `qualified_candidates_in_top_10`,
`must_have_check`, `requirement_details`, `score_breakdown`, `matched_sections`,
`experience_years`, `education`, and `latency_seconds`.

## Deliverables

| Deliverable | Status | Location |
| --- | --- | --- |
| Complete RAG implementation | Complete | `src/resume_rag.py`, `src/job_matcher.py` |
| Dataset: 30+ resumes, 5+ JDs | Complete | `resumes/`, `job_descriptions/` |
| Jupyter notebook with experimentation and analysis | Complete and executed | `notebooks/evaluation.ipynb` |
| Performance metrics: retrieval accuracy, latency | Complete | `outputs/evaluation_metrics.csv`, `outputs/final_evaluation_report.csv`, notebook |
| Demo video (3–4 minutes) | Complete — silent, captioned | `demo/Resume_RAG_Demo.mp4` (3:16) |

## Submission package

`Resume-RAG-Submission.zip` is a clean package containing the code, dataset,
notebook, outputs, documentation, demo video, and the built ChromaDB store. It
excludes `.venv`, `__pycache__`, and notebook checkpoints.

## Final verified results

| Job | Qualified in top 10 | Qualified candidate(s) |
| --- | ---: | --- |
| Python Backend Engineer | 0 | — |
| Machine Learning Engineer | 1 | Sana Khan |
| Production Engineer | 1 | Kavya Menon |
| Talent Acquisition Specialist | 0 | — |
| Cybersecurity Analyst | 0 | — |
| **Total** | **2** | |

Average metrics: Precision@10 = 0.44, Recall@10 = 0.733, retrieval accuracy =
44.0%, average latency ≈ 0.30 seconds per job after model load.

## Verification commands

```powershell
# Rebuild the vector database
.\.venv\Scripts\python.exe .\src\resume_rag.py

# Run all five job descriptions
Get-ChildItem .\job_descriptions\job_*.txt | ForEach-Object {
    .\.venv\Scripts\python.exe .\src\job_matcher.py $_.FullName
}

# Re-execute the notebook
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace .\notebooks\evaluation.ipynb

# Check the final artifacts
Get-Content .\outputs\evaluation_metrics.csv
Get-Content .\outputs\qualified_candidates.csv
```

The notebook ends with six automated validation checks; the latest execution
passed all six with zero errors.

## Notes

- All resumes are synthetic. Do not use this project for real hiring decisions.
- `demo/Resume_RAG_Demo.mp4` is a silent captioned screencast rendered from the
  real project files and real command output. `DEMO_SCRIPT.md` is available if
  you want to record a narrated live version instead.
- The original notebook is retained at `notebooks/archive/Untitled.ipynb`.
