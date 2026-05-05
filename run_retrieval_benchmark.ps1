param(
    [string]$Pdf = "review_on_tap.pdf",
    [string]$Question = "Noi dung chinh la gi?",
    [string[]]$References = @("Giay phep phan mem, Linux, AWS, Socket Python, bat dong bo va dong bo"),
    [string]$OllamaModel = "qwen2.5:0.5b" # Dung 0.5b cho nhe RAM
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "--- RUNNING RETRIEVAL BENCHMARK (Table III) ---" -ForegroundColor Cyan
Write-Host "Model: $OllamaModel"

& $VenvPython tools/retrieval_benchmark_auto.py `
    --pdf $Pdf `
    --question $Question `
    --references $References `
    --ollama-model $OllamaModel

$ModelSafe = $OllamaModel -replace ':', '_'
Write-Host ""
Write-Host "DONE! Check report at: documentation/retrieval_benchmark_$ModelSafe.md" -ForegroundColor Green
