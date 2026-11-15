
import os
import re
import csv
import hashlib
from pathlib import Path

import pymupdf as fitz
import chromadb
import spacy
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESUME_DIR = BASE_DIR / "resumes"
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "chroma_db"
MANIFEST_FILE = BASE_DIR / "resume_manifest.csv"

COLLECTION_NAME = "resume_collection"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 30

DATA_DIR.mkdir(exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ==========================================
# METADATA FROM MANIFEST
# ==========================================

def load_manifest():

    manifest = {}

    if not MANIFEST_FILE.exists():
        print("Warning: Manifest not found.")
        return manifest

    with open(MANIFEST_FILE, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:
            manifest[row["resume_file"]] = row

    return manifest


# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

def extract_text(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page in document:
        pages.append(page.get_text("text"))

    document.close()

    return "\n".join(pages)


# ==========================================
# METADATA EXTRACTION
# ==========================================

def extract_metadata(text, filename, manifest):

    record = manifest.get(filename, {})

    doc = nlp(text[:100000])

    name = record.get("candidate_name", "").strip()

    if not name:
        name = next(
            (ent.text for ent in doc.ents if ent.label_ == "PERSON"),
            Path(filename).stem
        )

    skills = record.get("skills", "").strip()

    if not skills:
        skills = extract_skills(text)

    education = record.get("education", "").strip()

    if not education:
        education = extract_education(text)

    experience = record.get("experience_years", "").strip()

    if not experience:
        experience = extract_experience(text)

    try:
        experience = float(experience)
    except (ValueError, TypeError):
        experience = -1.0

    return {
        "candidate_name": name,
        "skills": skills,
        "experience_years": experience,
        "education": education
    }


def extract_skills(text):

    skill_list = [
        "Python", "Java", "SQL", "Machine Learning",
        "TensorFlow", "PyTorch", "Django", "Flask",
        "FastAPI", "AWS", "Docker", "Kubernetes",
        "Pandas", "NumPy", "Power BI", "Excel",
        "AutoCAD", "SolidWorks", "Six Sigma",
        "Recruitment", "Linux", "NLP", "PostgreSQL",
        "React", "Git", "SPC", "FMEA", "SAP",
        "Selenium", "Scikit-learn", "PyTest"
    ]

    found = []

    text_lower = text.lower()

    for skill in skill_list:
        if skill.lower() in text_lower:
            found.append(skill)

    return ", ".join(found)


def extract_education(text):

    patterns = [
        r"(?:B\.Tech|B\.E\.|B\.Sc\.|B\.Com|M\.Tech|M\.Sc\.|MCA|MBA|Ph\.D\.|Diploma)[^\n]{0,100}"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)

        if matches:
            return " | ".join(matches[:3])

    return "Not specified"


def extract_experience(text):

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s+years?\s+of\s+experience",
        r"experience\s+of\s+(\d+(?:\.\d+)?)\+?\s+years?"
    ]

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return float(match.group(1))

    return -1.0


# ==========================================
# INTELLIGENT SECTION DETECTION
# ==========================================

SECTION_NAMES = {
    "summary": ["professional summary", "summary", "profile", "objective"],
    "skills": ["skills", "technical skills", "core competencies"],
    "experience": ["experience", "work experience", "professional experience"],
    "education": ["education", "academic background", "qualifications"],
    "projects": ["projects", "key projects", "personal projects"],
    "certifications": ["certifications", "certificates"]
}


def detect_section(line):

    cleaned = re.sub(r"[^a-zA-Z ]", "", line).strip().lower()

    for section, names in SECTION_NAMES.items():

        if cleaned in names:
            return section

    return None


def split_sections(text):

    sections = {}
    current_section = "general"

    for line in text.splitlines():

        section = detect_section(line)

        if section:
            current_section = section
            sections.setdefault(current_section, [])
            continue

        if line.strip():
            sections.setdefault(current_section, []).append(line.strip())

    return {
        section: "\n".join(lines)
        for section, lines in sections.items()
    }


# ==========================================
# TEXT CHUNKING
# ==========================================

def chunk_text(text, chunk_size=CHUNK_SIZE,
               overlap=CHUNK_OVERLAP):

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_document_chunks(text):

    sections = split_sections(text)

    results = []

    for section, content in sections.items():

        if not content.strip():
            continue

        chunks = chunk_text(content)

        for index, chunk in enumerate(chunks):

            results.append({
                "section": section,
                "text": chunk,
                "chunk_index": index
            })

    return results


# ==========================================
# RESUME PROCESSING
# ==========================================

def process_resume(pdf_path, manifest):

    filename = pdf_path.name

    print(f"\nProcessing: {filename}")

    text = extract_text(pdf_path)

    if not text.strip():
        print("No extractable text. Skipping.")
        return 0

    metadata = extract_metadata(
        text,
        filename,
        manifest
    )

    chunks = create_document_chunks(text)

    if not chunks:
        print("No chunks generated. Skipping.")
        return 0

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:

        unique_text = (
            f"{filename}|{chunk['section']}|"
            f"{chunk['chunk_index']}|{chunk['text']}"
        )

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

        documents.append(chunk["text"])

        metadatas.append({
            "candidate_name": metadata["candidate_name"],
            "resume_path": f"resumes/{filename}",
            "section": chunk["section"],
            "skills": metadata["skills"],
            "experience_years": metadata["experience_years"],
            "education": metadata["education"],
            "chunk_index": chunk["chunk_index"]
        })

    # Generate embeddings
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    # Remove old chunks for this resume before replacing
    collection.delete(
        where={"resume_path": f"resumes/{filename}"}
    )

    # Store in ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Candidate: {metadata['candidate_name']}")
    print(f"Chunks created: {len(chunks)}")
    print("Stored successfully.")

    return len(chunks)


# ==========================================
# MAIN PIPELINE
# ==========================================

def main():

    if not RESUME_DIR.exists():
        print("Resume directory does not exist.")
        return

    pdf_files = sorted(RESUME_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF resumes found!")
        return

    print("\n================================")
    print("RESUME RAG PROCESSING STARTED")
    print("================================")

    print(f"Total resumes: {len(pdf_files)}")

    manifest = load_manifest()

    total_chunks = 0
    processed = 0

    for pdf_path in pdf_files:

        try:

            count = process_resume(
                pdf_path,
                manifest
            )

            if count > 0:
                processed += 1
                total_chunks += count

        except Exception as error:

            print(f"Error processing {pdf_path.name}: {error}")

    print("\n================================")
    print("PROCESSING COMPLETED")
    print("================================")

    print(f"Resumes processed: {processed}")
    print(f"Total chunks: {total_chunks}")
    print(f"Database records: {collection.count()}")

    print(f"Database location: {DB_DIR}")


if __name__ == "__main__":
    main()
