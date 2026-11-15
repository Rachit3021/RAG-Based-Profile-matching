"""Print a concise demo summary for one job result file."""

import json
import sys
from pathlib import Path


def clean_excerpt(text):
    """Normalize common PDF punctuation for readable terminal output."""
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python demo/show_results.py <job-results.json>")

    path = Path(sys.argv[1])

    if not path.exists():
        raise SystemExit(f"Result file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    matches = data.get("top_matches", [])[:5]
    qualified = [
        candidate
        for candidate in data.get("top_matches", [])
        if candidate.get("must_have_check", {}).get("passed")
    ]

    print()
    print("=" * 78)
    print("JOB:", data.get("job_description"))
    print("REQUIRED EXPERIENCE:", data.get("detected_required_experience"), "years")
    print("QUALIFIED CANDIDATES IN TOP 10:", len(qualified))
    print("=" * 78)

    for index, candidate in enumerate(matches, start=1):
        check = candidate.get("must_have_check", {})
        status = "QUALIFIED" if check.get("passed") else "REJECTED"
        missing = check.get("missing_requirements", [])
        evidence = check.get("matched_required_skills", [])

        print(
            f"{index}. {candidate['candidate_name']:<20} "
            f"score={candidate['match_score']:<6} "
            f"experience={candidate['experience_years']} yrs  {status}"
        )

        if evidence:
            print("   evidence:", ", ".join(evidence))
        else:
            print("   evidence: none")

        if missing:
            print("   missing: ", "; ".join(missing))
        else:
            print("   missing:  none")

        excerpts = candidate.get("relevant_excerpts", [])
        if excerpts:
            print("   excerpt: ", clean_excerpt(excerpts[0])[:160])

        print()

    if qualified:
        print("QUALIFIED:", ", ".join(c["candidate_name"] for c in qualified))
    else:
        print("QUALIFIED: none")

    print()


if __name__ == "__main__":
    main()
