#!/bin/bash

# Step 1: Install required dependencies
sudo apt-get update -y
sudo apt-get install -y pciutils 

# Step 2: Download and install Ollama using the official install script
curl -fsSL https://ollama.com/install.sh | sh

# Step 3: Verify installation
#ollama version

# Step 4: Start the Ollama server
ollama serve &

# Step 5: Pull a model to use with Ollama
# Replace `gemma2` with the model name you need
#ollama pull gemma2
#ollama pull phi3:14b
#ollama pull llama3.2:7b

ollama pull llama3.2:latest
ollama pull qwen2:1.5b
ollama pull llama3.2:7b
