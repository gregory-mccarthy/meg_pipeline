#!/bin/bash

set -e

REPO="/Users/gm33/meg_pipeline"

if [ -z "$1" ]; then
    echo "Usage: $0 \"commit message\""
    exit 1
fi

COMMIT_MSG="$1"

cd "$REPO"

echo "--------------------------------"
echo "Git status:"
echo "--------------------------------"
git status --short

echo "--------------------------------"
echo "Adding updated files..."
echo "--------------------------------"
git add .

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

echo "--------------------------------"
echo "Committing..."
echo "--------------------------------"
git commit -m "$COMMIT_MSG"

BRANCH=$(git branch --show-current)

echo "--------------------------------"
echo "Pushing to GitHub (origin/$BRANCH)..."
echo "--------------------------------"
git push origin "$BRANCH"

echo "--------------------------------"
echo "Done."
echo "--------------------------------"