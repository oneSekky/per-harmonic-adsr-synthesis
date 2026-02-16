#!/bin/bash
# Script to create GitHub repository via API

echo "Creating GitHub repository: per-harmonic-adsr-synthesis"
echo "========================================================="
echo ""

# Check if user has GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  GITHUB_TOKEN environment variable not set."
    echo ""
    echo "Option 1: Create via GitHub Web Interface (Recommended - 2 minutes)"
    echo "  1. Go to: https://github.com/new"
    echo "  2. Repository name: per-harmonic-adsr-synthesis"
    echo "  3. Description: Parametric audio synthesizer enabling pitch-independent timbre reproduction through per-harmonic ADSR envelope decomposition"
    echo "  4. Set to Public"
    echo "  5. DO NOT check 'Add README', 'Add .gitignore', or 'Add license'"
    echo "  6. Click 'Create repository'"
    echo ""
    echo "Option 2: Create via Personal Access Token"
    echo "  1. Go to: https://github.com/settings/tokens/new"
    echo "  2. Generate a token with 'repo' scope"
    echo "  3. Run: export GITHUB_TOKEN='your_token_here'"
    echo "  4. Run this script again"
    echo ""
    echo "After creating the repository, run:"
    echo "  cd /home/sekky/Downloads/per-harmonic-adsr-synthesis"
    echo "  git push -u origin main"
    exit 1
fi

# Create repository using GitHub API
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d '{
    "name": "per-harmonic-adsr-synthesis",
    "description": "Parametric audio synthesizer enabling pitch-independent timbre reproduction through per-harmonic ADSR envelope decomposition",
    "homepage": "https://github.com/oneSekky/per-harmonic-adsr-synthesis",
    "private": false,
    "has_issues": true,
    "has_projects": true,
    "has_wiki": false
  }'

echo ""
echo ""
echo "✅ Repository created! Now pushing code..."
echo ""

cd /home/sekky/Downloads/per-harmonic-adsr-synthesis
git push -u origin main

echo ""
echo "✅ Code pushed to GitHub!"
echo ""
echo "View your repository at:"
echo "https://github.com/oneSekky/per-harmonic-adsr-synthesis"
