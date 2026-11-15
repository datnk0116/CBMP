#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "[ERROR] Not inside the CBMP Git repository."
  exit 2
}
cd "$ROOT"

EXPECTED_BRANCH="fix/assignment2-compliance"
BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "[ERROR] Expected branch '$EXPECTED_BRANCH', got '$BRANCH'."
  exit 2
fi

echo "== CBMP submission preflight =="
echo "Repository: $ROOT"
echo "Branch:     $BRANCH"
echo "Commit:     $(git rev-parse HEAD)"

PYTHON="${PYTHON:-python3}"
"$PYTHON" - <<'PY'
import sys
print(f"Python: {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required.")
PY

echo
echo "== 1/5 Whitespace check =="
git diff --check

echo
echo "== 2/5 Dataset integrity =="
"$PYTHON" scripts/check_datasets.py

echo
echo "== 3/5 Automated tests (excluding explicitly slow tests) =="
"$PYTHON" -m pytest -q -m "not slow"

echo
echo "== 4/5 Report build =="
if command -v latexmk >/dev/null 2>&1; then
  (cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex)
elif command -v pdflatex >/dev/null 2>&1; then
  (
    cd report
    pdflatex -interaction=nonstopmode -halt-on-error report.tex
    pdflatex -interaction=nonstopmode -halt-on-error report.tex
  )
else
  echo "[WARN] latexmk/pdflatex not installed; report build skipped."
fi

echo
echo "== 5/5 Official benchmark archive =="
if [[ ! -s results/final/runs.jsonl ]]; then
  echo "[BLOCKER] results/final/runs.jsonl is missing or empty."
  echo "The 44 exploratory records in results_v2 are dirty-source data and must not be copied here."
  echo "Run the configured experiments from this clean commit before claiming a reproducible final archive."
  exit 3
fi

"$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("results/final/runs.jsonl")
count = 0
for line_no, line in enumerate(p.read_text().splitlines(), 1):
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSONL at {p}:{line_no}: {exc}")
    if not isinstance(record, dict):
        raise SystemExit(f"Expected JSON object at {p}:{line_no}")
    count += 1
if count == 0:
    raise SystemExit("Official JSONL archive contains no records.")
env = Path("results/final/env.json")
if not env.is_file() or not env.stat().st_size:
    raise SystemExit("Official results/final/env.json is missing or empty.")
print(f"[OK] {count} valid JSONL records and environment metadata found.")
PY

echo
echo "[PASS] Preflight completed."
git status --short
