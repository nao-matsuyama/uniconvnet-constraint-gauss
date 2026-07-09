#!/usr/bin/env bash
set -e

# Initialize git and lfs, make first commit
if [ ! -d .git ]; then
  git init
  git lfs install
  git add .gitattributes
  git add .
  git commit -m "Initial scaffold for versioned workspace"
  echo "Repository initialized and initial commit made."
else
  echo "Git already initialized."
fi
