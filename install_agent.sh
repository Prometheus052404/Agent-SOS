#!/bin/bash
# Xv6 Pedagogical Agent - Quick Install Script
# Usage: bash install_agent.sh [GROQ_API_KEY]

set -e

echo "🎓 Xv6 Pedagogical Agent Installer"
echo "==================================="

# Check if we're in an xv6 directory
if [ ! -f "Makefile" ] || ! grep -q "xv6" Makefile 2>/dev/null; then
    echo "⚠️  Warning: This doesn't look like an xv6 directory"
    echo "   Make sure you're in your xv6-public folder"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.9+ required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install --quiet --upgrade pip
pip install --quiet tree-sitter tree-sitter-c networkx chromadb \
    sentence-transformers watchdog click rich pyyaml \
    python-dotenv groq openai anthropic

echo "✓ Dependencies installed"

# Check/create .xv6_agent directory
if [ ! -d ".xv6_agent" ]; then
    echo "❌ .xv6_agent directory not found!"
    echo "   Copy it from the source: cp -r /path/to/.xv6_agent ."
    exit 1
fi

# Setup API key
if [ -n "$1" ]; then
    echo "GROQ_API_KEY=$1" > .xv6_agent/.env
    echo "✓ API key configured"
else
    if [ ! -f ".xv6_agent/.env" ]; then
        echo ""
        read -p "🔑 Enter your GROQ API key: " API_KEY
        if [ -n "$API_KEY" ]; then
            echo "GROQ_API_KEY=$API_KEY" > .xv6_agent/.env
            echo "✓ API key saved"
        else
            echo "⚠️  No API key provided. Add it later to .xv6_agent/.env"
        fi
    else
        echo "✓ API key already configured"
    fi
fi

# Make agent executable
chmod +x agent 2>/dev/null || true

# Initialize
echo ""
echo "🔧 Initializing agent..."
./agent init

echo ""
echo "========================================="
echo "✅ Installation Complete!"
echo "========================================="
echo ""
echo "To start using:"
echo "  source .venv/bin/activate"
echo "  ./agent help \"your question\""
echo ""
echo "Quick test:"
echo "  ./agent status"
echo ""
