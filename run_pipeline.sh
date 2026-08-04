#!/bin/bash

# Set path to include local static ffmpeg
export PATH="/home/linuxlite/.gemini/antigravity/scratch/ai_video_automation/bin:$PATH"

# Set API Keys and Config (You can change the RSS feed URL if needed)
export NITTER_RSS_URL="https://nitter.poast.org/SpongeBob/rss"
export NVIDIA_API_KEY="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"

echo "=========================================="
echo "🎬 Starting AI Video Automation Pipeline"
echo "=========================================="

echo "[1/5] Running Downloader Agent..."
python3 agents/downloader_agent.py || { echo "Downloader failed"; exit 1; }

echo "[2/5] Running Video Analysis Agent (This might take a while)..."
python3 agents/video_analysis_agent.py || { echo "Analysis failed"; exit 1; }

echo "[3/5] Running Script Writer Agent (NVIDIA Nemotron LLM)..."
python3 agents/script_writer_agent.py || { echo "Script Writer failed"; exit 1; }

echo "[4/5] Running Voice Generation Agent (Piper TTS)..."
python3 agents/voice_generation_agent.py || { echo "Voice Generation failed"; exit 1; }

echo "[5/5] Running Smart Editing Agent (FFmpeg)..."
python3 agents/smart_editing_agent.py || { echo "Smart Editing failed"; exit 1; }

echo "=========================================="
echo "✅ Done! Check the 'exports' folder for your final video."
echo "=========================================="
