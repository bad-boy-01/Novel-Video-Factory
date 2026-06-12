#!/bin/bash
echo "Starting Novel Video Factory Kaggle Setup..."

# 1. Install Ollama
echo "Installing prerequisites..."
apt-get update && apt-get install -y zstd

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
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install xformers

echo "Setup complete! You can now run the pipeline."
