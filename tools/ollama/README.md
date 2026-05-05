# Ollama Setup

This directory contains Ollama binaries and models (ignored by git due to large size).

## Download Ollama

Download Ollama for Windows from: https://ollama.com/download

Or extract from the zip file if you have it locally.

## Required Models

After installing Ollama, download the required models:

```bash
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:7b
```

## Directory Structure

```
tools/ollama/
├── ollama.exe          (ignored)
├── lib/                (ignored)
├── models/             (ignored)
└── README.md           (this file)
```
