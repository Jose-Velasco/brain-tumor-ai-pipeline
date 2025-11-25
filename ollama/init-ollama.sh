#!/usr/bin/env bash
set -e

# start ollama server in background
ollama serve &

# wait until API is up
until curl -sSf http://localhost:11434/api/tags > /dev/null; do
  echo "Waiting for ollama..."
  sleep 2
done

# ollama pull gemma3:270m
ollama pull gemma3:4b-it-qat

# keep server in foreground (wait for background ollama)
wait