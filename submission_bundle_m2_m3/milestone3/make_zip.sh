#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./make_zip.sh <LastName> <FirstName>
# Example:
#   ./make_zip.sh Doe Jane

if [[ $# -ne 2 ]]; then
  echo "Usage: ./make_zip.sh <LastName> <FirstName>"
  exit 1
fi

LAST_NAME="$1"
FIRST_NAME="$2"
ZIP_NAME="CS5590_Grad_M3_${LAST_NAME}_${FIRST_NAME}.zip"

if [[ ! -f "paper.pdf" ]]; then
  echo "Missing required file: paper.pdf"
  exit 1
fi

if [[ ! -f "demo_video.mp4" && ! -f "demo_link.txt" ]]; then
  echo "Missing demo deliverable: add demo_video.mp4 or demo_link.txt"
  exit 1
fi

rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" \
  code \
  results \
  README.md \
  README_M3.md \
  M2_M3_RESULTS_AND_PAPER_GUIDE.md \
  M3_TEST_REPORT.md \
  SUBMISSION_CHECKLIST.md \
  requirements.txt \
  paper.pdf \
  $( [[ -f demo_video.mp4 ]] && echo demo_video.mp4 || echo demo_link.txt )

echo "Created: $ZIP_NAME"
