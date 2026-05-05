param(
    [string]$Pdf = "review_on_tap.pdf",
    [string]$OllamaModel = "qwen2.5:0.5b" # Dùng 0.5b cho nhẹ RAM
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "🚀 Chạy Benchmark RAG vs CoRAG (Table V)..." -ForegroundColor Cyan
Write-Host "Model: $OllamaModel"

& $VenvPython tools/corag_benchmark_auto.py --pdf $Pdf --ollama-model $OllamaModel

Write-Host "`n✅ Hoàn tất! Xem báo cáo Table V và VI tại: documentation/corag_benchmark_$($OllamaModel -replace ':', '_').md" -ForegroundColor Green
