#!/bin/bash
echo "Starting Novel Video Factory Kaggle Setup..."

# 1. Install Ollama
echo "Installing prerequisites..."
apt-get update && apt-get install -y zstd espeak-ng imagemagick

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama server in the background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
sleep 5 # wait for server to start

# 3. Pull required LLM model
echo "Pulling Qwen2.5:7b model..."
ollama pull qwen2.5:7b

# 4. Install python dependencies
echo "Installing Python dependencies (Ignoring non-critical Kaggle conflicts)..."
pip install --quiet --no-warn-conflicts -r requirements.txt
pip install --quiet xformers

echo "Setup complete! You can now run the pipeline."
echo "NOTE: You may see 'Dependency Errors' related to cudf or dask; these are safe to ignore as they don't affect this project."
