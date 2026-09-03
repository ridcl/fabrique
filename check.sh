#!/bin/bash
# Lint, format and type-check.  Pass --fix to apply safe fixes and reformat in
# place; without it nothing is modified, so it is safe to run in CI.
set -e

if [[ "$1" == "--fix" ]]; then
    ruff check --fix .
    ruff format .
else
    ruff check .
    ruff format --check .
fi

ty check .
