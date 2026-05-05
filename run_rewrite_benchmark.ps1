param(
    [string]$Pdf = "review_on_tap.pdf",
    [string]$OllamaModel = "qwen2.5:0.5b" # Dung 0.5b cho nhe RAM
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "🚀 Chạy Benchmark Query Rewriting..." -ForegroundColor Cyan
Write-Host "Model: $OllamaModel"

& $VenvPython tools/rewrite_benchmark_auto.py --pdf $Pdf --ollama-model $OllamaModel

Write-Host "`n✅ Hoàn tất! Xem báo cáo tại thư mục documentation/" -ForegroundColor Green
