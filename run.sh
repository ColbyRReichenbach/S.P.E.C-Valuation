#!/bin/bash
# ====================================
# S.P.E.C. Valuation Engine - Run Script
# ====================================

set -e

echo "🏠 S.P.E.C. Valuation Engine"
echo "============================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "   Edit .env to add your OPENAI_API_KEY for AI features."
fi

echo ""
echo "🚀 Starting Streamlit server..."
echo "   Dashboard will open at http://localhost:8501"
echo ""

# Run Streamlit
streamlit run app.py --server.headless=true
