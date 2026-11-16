"""Print the evaluation summary from the saved metrics CSV."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "evaluation_metrics.csv"
QUALIFIED = ROOT / "outputs" / "qualified_candidates.csv"


def main():
    rows = list(csv.DictReader(METRICS.open(encoding="utf-8")))
    qualified = list(csv.DictReader(QUALIFIED.open(encoding="utf-8")))

    print()
    print("Job                                  P@10   R@10   Acc%   Qualified")
    print("-" * 68)
    for row in rows:
        name = row["Job"].replace("job_", "")[:34]
        print(
            f"{name:<36}"
            f"{float(row['Precision@10']):>5.2f}"
            f"{float(row['Recall@10']):>7.3f}"
            f"{float(row['Retrieval Accuracy (%)']):>7.1f}"
            f"{int(row['Qualified Candidates']):>10}"
        )

    print("-" * 68)
    avg_p = sum(float(r["Precision@10"]) for r in rows) / len(rows)
    avg_r = sum(float(r["Recall@10"]) for r in rows) / len(rows)
    avg_acc = sum(float(r["Retrieval Accuracy (%)"]) for r in rows) / len(rows)
    avg_lat = sum(float(r["Latency (seconds)"]) for r in rows) / len(rows)
    total = sum(int(r["Qualified Candidates"]) for r in rows)

    print(f"Average Precision@10 : {avg_p:.3f}")
    print(f"Average Recall@10    : {avg_r:.3f}")
    print(f"Average Accuracy     : {avg_acc:.1f}%")
    print(f"Average Latency      : {avg_lat:.2f}s")
    print(f"Qualified candidates : {total}")
    print()
    for row in qualified:
        print(f"  {row['Candidate Name']} - {row['Match Score']} ({row['Job']})")
    print()


if __name__ == "__main__":
    main()
