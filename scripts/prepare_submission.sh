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

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found."
  exit 2
fi

PYTHON="${PYTHON:-python3}"
"$PYTHON" - <<'PY'
import sys
print(f"Python: {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required.")
PY

echo
echo "== 1/5 Repository whitespace check =="
git diff --check

echo
echo "== 2/5 Dataset integrity check =="
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
echo "== 5/5 Final evidence archive check =="
if [[ ! -s results/final/runs.jsonl ]]; then
  echo "[BLOCKER] results/final/runs.jsonl is missing or empty."
  echo "Do NOT copy results_v2/runs.jsonl into results/final: its metadata marks the source as dirty."
else
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("results/final/runs.jsonl")
records = []
for line_no, line in enumerate(p.read_text().splitlines(), 1):
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSONL at {p}:{line_no}: {exc}")
    if not isinstance(row, dict):
        raise SystemExit(f"Expected JSON object at {p}:{line_no}")
    records.append(row)
if not records:
    raise SystemExit("Final JSONL contains no records.")
print(f"Valid JSONL records: {len(records)}")
PY
  if [[ ! -s results/final/env.json ]]; then
    echo "[BLOCKER] results/final/env.json is missing or empty."
  fi
fi

echo
echo "== Summary =="
echo "Checks completed. Review any [WARN]/[BLOCKER] above."
echo "Current git status:"
git status --short
