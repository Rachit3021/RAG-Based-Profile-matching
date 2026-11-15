
import os
import re
import json
import shutil
import argparse
import time

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

COLLECTION_NAME = "resume_collection"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SKILL KEYWORDS AND ALIASES
# ============================================================

SKILL_KEYWORDS = [
    "python", "java", "c++", "javascript", "typescript",
    "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "artificial intelligence",
    "data analysis", "data science", "pandas", "numpy",
    "tensorflow", "pytorch", "scikit-learn",
    "aws", "azure", "gcp", "docker", "kubernetes",
    "linux", "networking", "git", "rest api",
    "fastapi", "django", "flask",
    "cybersecurity", "network security", "penetration testing",
    "siem", "splunk", "incident response",
    "firewall", "vulnerability assessment",
    "solidworks", "autocad", "catia",
    "manufacturing", "production planning",
    "quality control", "lean manufacturing",
    "six sigma", "hvac", "automotive",
    "recruitment", "talent acquisition",
    "human resources", "hr operations",
    "employee relations", "payroll",
    "communication", "project management",
    "problem solving", "leadership",
    "root cause analysis", "continuous improvement",
    "candidate sourcing", "interview coordination",
    "stakeholder management", "applicant tracking system",
    "redis", "restful api"
]


# Each requirement group uses OR logic between its alternatives.
# Different groups are combined using AND logic.

REQUIREMENT_PROFILES = {

    "python": [
        {
            "label": "Python programming",
            "any": ["python"]
        },
        {
            "label": "REST API development",
            "any": [
                "rest api",
                "restful api",
                "rest apis",
                "building rest apis",
                "built rest apis"
            ]
        },
        {
            "label": "API framework (Django, Flask or FastAPI)",
            "any": ["django", "flask", "fastapi"]
        },
        {
            "label": "Relational database",
            "any": ["postgresql", "mysql", "sql"]
        },
        {
            "label": "Docker",
            "any": ["docker"]
        },
        {
            "label": "Production deployment",
            "any": [
                "production deployment",
                "deployment",
                "deployed",
                "deploying",
                "production environment"
            ]
        }
    ],

    "machine_learning": [
        {
            "label": "Machine learning",
            "any": ["machine learning"]
        },
        {
            "label": "Python programming",
            "any": ["python"]
        },
        {
            "label": "Deep learning framework (PyTorch or TensorFlow)",
            "any": ["pytorch", "tensorflow"]
        },
        {
            "label": "ML model deployment or maintenance",
            "any": [
                "model deployment",
                "deploying models",
                "deployed models",
                "maintaining models",
                "model monitoring",
                "ml model deployment",
                "ml model maintenance",
                "deployed nlp classification models",
                "deployed ml models",
                "monitored production drift"
            ]
        }
    ],

    "production": [
        {
            "label": "Manufacturing or production engineering",
            "any": ["manufacturing", "production engineering"]
        },
        {
            "label": "Lean manufacturing or Six Sigma",
            "any": ["lean manufacturing", "six sigma"]
        },
        {
            "label": "Root cause analysis",
            "any": ["root cause analysis", "rca"]
        },
        {
            "label": "Continuous improvement",
            "any": ["continuous improvement", "kaizen"]
        },
        {
            "label": "Engineering qualification",
            "any": [
                "b.tech",
                "btech",
                "b.e.",
                "bachelor of engineering",
                "bachelor's degree in engineering",
                "engineering degree",
                "mechanical engineering",
                "industrial engineering",
                "production engineering"
            ]
        }
    ],

    "hr": [
        {
            "label": "Recruitment or talent acquisition",
            "any": ["recruitment", "talent acquisition"]
        },
        {
            "label": "Candidate sourcing",
            "any": [
                "candidate sourcing",
                "sourcing candidates",
                "talent sourcing",
                "sourced candidates",
                "sourcing"
            ]
        },
        {
            "label": "Interview coordination",
            "any": [
                "interview coordination",
                "coordinated interviews",
                "interview scheduling",
                "scheduling interviews"
            ]
        },
        {
            "label": "ATS or recruitment tracking",
            "any": [
                "ats",
                "applicant tracking system",
                "recruitment tracking",
                "applicant tracking"
            ]
        },
        {
            "label": "Stakeholder communication",
            "any": [
                "stakeholder communication",
                "stakeholder management",
                "communication",
                "communicated with"
            ]
        }
    ],

    "cybersecurity": [
        {
            "label": "Cybersecurity or security operations",
            "any": [
                "cybersecurity",
                "security operations",
                "information security"
            ]
        },
        {
            "label": "SIEM monitoring",
            "any": ["siem", "security information and event management"]
        },
        {
            "label": "Security alert investigation",
            "any": [
                "alert investigation",
                "security alert investigation",
                "investigated security alerts",
                "security monitoring"
            ]
        },
        {
            "label": "Incident response",
            "any": ["incident response"]
        },
        {
            "label": "Linux",
            "any": ["linux"]
        },
        {
            "label": "Networking",
            "any": ["networking", "network administration", "tcp/ip"]
        }
    ]
}


# ============================================================
# LOAD MODEL AND VECTOR DATABASE
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Resume database loaded.")
print("Database records:", collection.count())


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_term(text, term):
    """
    Match a term without accidentally matching parts
    of unrelated words.

    The final word may use a plural form, so "REST API"
    matches "REST APIs" while unrelated words are still rejected.
    """

    text = clean_text(text)
    term = clean_text(term)

    words = term.split()

    if not words:
        return False

    prefix = r"\s+".join(
        re.escape(word) for word in words[:-1]
    )

    last_word = re.escape(words[-1])

    if prefix:
        pattern = (
            r"(?<![a-z0-9])"
            + prefix
            + r"\s+"
            + last_word
            + r"(?:s|es)?"
            + r"(?![a-z0-9])"
        )
    else:
        pattern = (
            r"(?<![a-z0-9])"
            + last_word
            + r"(?:s|es)?"
            + r"(?![a-z0-9])"
        )

    return re.search(pattern, text) is not None


def extract_skills(text):
    text = clean_text(text)

    found = []

    for skill in SKILL_KEYWORDS:
        if contains_term(text, skill):
            found.append(skill)

    return sorted(set(found))


# ============================================================
# EXPERIENCE EXTRACTION
# ============================================================

def extract_years(text):
    """
    Extract years of experience from resume text.
    """

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+experience",
        r"experience\s*[:\-]\s*(\d+(?:\.\d+)?)\s*\+?\s*years?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return float(match.group(1))

    return 0.0


def extract_required_experience(jd):

    patterns = [
        r"\b(?:minimum\s+of|at\s+least)\s+(\d+)\s*\+?\s*years?\b",
        r"\b(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?"
        r"(?:[a-z][a-z0-9+#./ -]{0,60}?)"
        r"(?=\bexperience\b|\bdevelopment\b|\bin\s+[a-z]|\bwith\b|[,.;\n]|$)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            jd,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    return None


# ============================================================
# JOB TYPE DETECTION
# ============================================================

def detect_job_type(jd):

    text = clean_text(jd)

    if "python backend engineer" in text:
        return "python"

    if "machine learning engineer" in text:
        return "machine_learning"

    if "production engineer" in text:
        return "production"

    if "talent acquisition" in text:
        return "hr"

    if "cybersecurity analyst" in text:
        return "cybersecurity"

    return "general"


# ============================================================
# REQUIREMENT EXTRACTION
# ============================================================

def extract_must_have_requirements(jd):

    job_type = detect_job_type(jd)

    if job_type in REQUIREMENT_PROFILES:
        return REQUIREMENT_PROFILES[job_type]

    # Generic fallback for other job descriptions.
    # Extract known skills from the must-have section.

    lines = jd.splitlines()

    collecting = False
    required_lines = []

    for line in lines:

        normalized = clean_text(line)

        if any(marker in normalized for marker in [
            "must-have",
            "must have",
            "required skills",
            "requirements",
            "mandatory"
        ]):

            collecting = True
            continue

        if collecting:

            if any(marker in normalized for marker in [
                "preferred",
                "nice to have",
                "good to have",
                "responsibilities",
                "desired"
            ]):
                break

            required_lines.append(line)

    skills = extract_skills(" ".join(required_lines))

    return [
        {
            "label": skill,
            "any": [skill]
        }
        for skill in skills
    ]


def extract_must_have_skills(jd):

    requirements = extract_must_have_requirements(jd)

    skills = []

    for requirement in requirements:
        skills.extend(requirement["any"])

    return sorted(set(skills))


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def retrieve_candidates(jd, top_k=TOP_K):

    """
    Retrieve resume chunks from ChromaDB.

    Chunks are grouped by resume_path.
    Candidate ranking happens after grouping.
    """

    jd_embedding = model.encode([jd]).tolist()

    results = collection.query(
        query_embeddings=jd_embedding,
        n_results=collection.count(),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    candidates = {}

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        metadata = metadata or {}

        resume_path = metadata.get(
            "resume_path",
            metadata.get("source", "unknown")
        )

        candidate_name = metadata.get(
            "candidate_name",
            metadata.get("name", "Unknown")
        )

        semantic_score = max(
            0.0,
            min(
                100.0,
                (1.0 - float(distance)) * 100.0
            )
        )

        if resume_path not in candidates:

            candidates[resume_path] = {
                "candidate_name": candidate_name,
                "resume_path": resume_path,
                "chunks": [],
                "semantic_scores": [],
                "metadata": metadata
            }

        candidates[resume_path]["chunks"].append(document)

        candidates[resume_path]["semantic_scores"].append(
            semantic_score
        )

    for candidate in candidates.values():

        candidate["semantic_score"] = max(
            candidate["semantic_scores"]
        )

        candidate["resume_text"] = "\n".join(
            candidate["chunks"]
        )

    return list(candidates.values())


# ============================================================
# MUST-HAVE REQUIREMENT CHECKING
# ============================================================

def check_must_have(
    candidate,
    requirements,
    required_experience
):

    """
    All requirement groups must pass.

    Within each group, at least one alternative must match.

    Example:
        Django OR Flask OR FastAPI

    This is one requirement group, not three.
    """

    resume_text = clean_text(
        candidate["resume_text"]
    )

    experience = candidate.get(
        "experience_years",
        0.0
    )

    experience_passed = (
        required_experience is None
        or experience >= required_experience
    )

    requirement_results = []
    missing_requirements = []

    for requirement in requirements:

        alternatives = requirement["any"]

        matched = [
            term for term in alternatives
            if contains_term(resume_text, term)
        ]

        passed = len(matched) > 0

        requirement_results.append({
            "requirement": requirement["label"],
            "passed": passed,
            "matched_evidence": matched,
            "alternatives": alternatives
        })

        if not passed:
            missing_requirements.append(
                requirement["label"]
            )

    if not experience_passed:

        missing_requirements.append(
            f"{required_experience}+ years of experience required"
        )

    passed = (
        experience_passed
        and all(
            item["passed"]
            for item in requirement_results
        )
    )

    matched_required_skills = []

    for result in requirement_results:
        matched_required_skills.extend(
            result["matched_evidence"]
        )

    return {

        "passed": passed,

        "missing_requirements": missing_requirements,

        "experience_passed": experience_passed,

        "required_experience": required_experience,

        "candidate_experience": experience,

        "required_skills": [
            requirement["label"]
            for requirement in requirements
        ],

        "matched_required_skills": sorted(
            set(matched_required_skills)
        ),

        "requirement_details": requirement_results
    }


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(
    candidate,
    jd,
    requirements,
    required_experience
):

    resume_text = clean_text(
        candidate["resume_text"]
    )

    jd_text = clean_text(jd)

    candidate_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        jd_text
    )

    matched_skills = [
        skill
        for skill in job_skills
        if skill in candidate_skills
    ]

    if job_skills:

        keyword_score = (
            len(matched_skills) / len(job_skills)
        ) * 100.0

    else:
        keyword_score = 0.0

    semantic_score = candidate["semantic_score"]

    experience = extract_years(resume_text)

    candidate["experience_years"] = experience

    if required_experience is None:

        experience_score = 100.0

    elif experience >= required_experience:

        experience_score = 100.0

    else:

        experience_score = (
            experience / required_experience
        ) * 100.0 if required_experience > 0 else 100.0

    must_have_check = check_must_have(
        candidate,
        requirements,
        required_experience
    )

    final_score = (
        semantic_score * 0.60
        + keyword_score * 0.30
        + experience_score * 0.10
    )

    # Candidates failing mandatory requirements
    # cannot receive a score above 59.

    if not must_have_check["passed"]:
        final_score = min(final_score, 59.0)

    candidate["match_score"] = round(
        max(0.0, min(100.0, final_score)),
        2
    )

    candidate["matched_skills"] = matched_skills

    candidate["must_have_check"] = must_have_check

    candidate["score_breakdown"] = {
        "semantic": round(semantic_score, 2),
        "keyword": round(keyword_score, 2),
        "experience": round(experience_score, 2)
    }

    candidate["relevant_excerpts"] = (
        candidate["chunks"][:3]
    )

    candidate["matched_sections"] = [
        section
        for section in [
            "summary",
            "skills",
            "experience",
            "education",
            "general"
        ]
        if any(
            section in clean_text(chunk)
            for chunk in candidate["chunks"]
        )
    ]

    if must_have_check["passed"]:

        status = "Passed all mandatory requirements"

    else:

        status = (
            "Missing: "
            + ", ".join(
                must_have_check["missing_requirements"]
            )
        )

    candidate["reasoning"] = (
        f"Semantic score: {semantic_score:.2f}%. "
        f"Keyword score: {keyword_score:.2f}%. "
        f"Experience score: {experience_score:.2f}%. "
        f"Matched skills: "
        f"{', '.join(matched_skills) or 'None'}. "
        f"Mandatory check: {status}."
    )

    return candidate


# ============================================================
# JOB MATCHING PIPELINE
# ============================================================

def match_resumes(job_description_path):

    start_time = time.time()

    with open(
        job_description_path,
        "r",
        encoding="utf-8"
    ) as file:

        jd = file.read()

    print("\nJob description loaded.")

    required_experience = extract_required_experience(jd)

    requirements = extract_must_have_requirements(jd)

    required_skills = extract_must_have_skills(jd)

    job_skills = extract_skills(jd)

    print("Detected job type:", detect_job_type(jd))

    print("Required experience:", required_experience)

    print("\nMandatory requirement groups:")

    for requirement in requirements:
        print("-", requirement["label"])

    candidates = retrieve_candidates(jd)

    print("\nCandidates retrieved:", len(candidates))

    scored_candidates = []

    for candidate in candidates:

        scored = score_candidate(
            candidate,
            jd,
            requirements,
            required_experience
        )

        scored_candidates.append(scored)

    # Qualified candidates are ranked first.

    scored_candidates.sort(
        key=lambda item: (
            item["must_have_check"]["passed"],
            item["match_score"]
        ),
        reverse=True
    )

    top_matches = []

    for candidate in scored_candidates[:TOP_K]:

        top_matches.append({

            "candidate_name": candidate["candidate_name"],

            "resume_path": candidate["resume_path"],

            "match_score": candidate["match_score"],

            "matched_skills": candidate["matched_skills"],

            "relevant_excerpts": candidate["relevant_excerpts"],

            "matched_sections": candidate["matched_sections"],

            "experience_years": candidate["experience_years"],

            "education": candidate["metadata"].get(
                "education",
                ""
            ),

            "must_have_check": candidate["must_have_check"],

            "score_breakdown": candidate["score_breakdown"],

            "reasoning": candidate["reasoning"]
        })

    qualified_matches = [
        candidate
        for candidate in top_matches
        if candidate["must_have_check"]["passed"]
    ]

    latency = round(
        time.time() - start_time,
        4
    )

    output = {

        "job_description": os.path.basename(
            job_description_path
        ),

        "detected_required_experience": required_experience,

        "detected_must_have_skills": required_skills,

        "detected_mandatory_requirements": requirements,

        "detected_job_skills": job_skills,

        "retrieved_candidates": len(candidates),

        "qualified_candidates_in_top_10": len(
            qualified_matches
        ),

        "top_matches": top_matches,

        "qualified_matches": qualified_matches,

        "latency_seconds": latency
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{os.path.splitext(os.path.basename(job_description_path))[0]}-results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Keep a copy of the most recent run for quick inspection.
    latest_path = os.path.join(
        OUTPUT_DIR,
        "matching_results.json"
    )

    shutil.copyfile(output_path, latest_path)

    print("\n========== MATCHING RESULTS ==========")

    for index, candidate in enumerate(
        top_matches,
        start=1
    ):

        print(
            f"{index}. {candidate['candidate_name']} | "
            f"Score: {candidate['match_score']} | "
            f"Must-have passed: "
            f"{candidate['must_have_check']['passed']}"
        )

    print(
        "\nQualified candidates in top 10:",
        len(qualified_matches)
    )

    print("\nLatency:", latency, "seconds")

    print("\nResults saved to:", output_path)

    return output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Resume RAG Job Matching System"
    )

    parser.add_argument(
        "job_description",
        help="Path to the job description text file"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.job_description):

        raise FileNotFoundError(
            f"Job description not found: {args.job_description}"
        )

    match_resumes(args.job_description)
