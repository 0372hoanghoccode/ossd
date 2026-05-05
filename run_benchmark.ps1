# =============================================================================
# run_benchmark.ps1 - Script chạy benchmark phân đoạn văn bản
# =============================================================================
# Cách dùng:
#   .\run_benchmark.ps1
#   .\run_benchmark.ps1 -Question "Câu hỏi của bạn"
#   .\run_benchmark.ps1 -Pdf "path/to/file.pdf" -Question "Câu hỏi" -References @("ref1", "ref2")
# =============================================================================

param(
    [string]$Pdf = "review_on_tap.pdf",
    [string]$Question = "",
    [string[]]$References = @(),
    [string]$Output = "",
    [string]$RetrievalMode = "hybrid",
    [string]$OllamaModel = "qwen2.5:7b",
    [int]$RetrieverK = 3,
    [int]$Repeats = 1
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "❌ Không tìm thấy Python venv tại: $VenvPython" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Pdf)) {
    Write-Host "❌ Không tìm thấy file PDF: $Pdf" -ForegroundColor Red
    exit 1
}

# Prompt for question if not provided
if (-not $Question) {
    $Question = Read-Host "Nhập câu hỏi để benchmark"
    if (-not $Question) {
        $Question = "Nội dung chính của tài liệu này là gì?"
        Write-Host "Sử dụng câu hỏi mặc định: $Question" -ForegroundColor Yellow
    }
}

# Auto-generate output filename from model name if not specified
if (-not $Output) {
    $modelSafe = $OllamaModel -replace '[^a-zA-Z0-9._-]', '_'
    $Output = "documentation/chunk_benchmark_$modelSafe.md"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK CHIEN LUOC PHAN DOAN VAN BAN" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PDF:    $Pdf"
Write-Host "  Question: $Question"
Write-Host "  Model:  $OllamaModel"
Write-Host "  Mode:   $RetrievalMode"
Write-Host "  Output: $Output"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$args_list = @(
    "tools/chunk_benchmark_auto.py",
    "--pdf", $Pdf,
    "--question", $Question,
    "--output", $Output,
    "--retrieval-mode", $RetrievalMode,
    "--ollama-model", $OllamaModel,
    "--retriever-k", $RetrieverK,
    "--repeats", $Repeats
)

if ($References.Count -gt 0) {
    $args_list += "--references"
    $args_list += $References
}

& $VenvPython $args_list

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Benchmark hoàn tất! Báo cáo:" -ForegroundColor Green
    Write-Host "   📄 $Output" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Benchmark thất bại (exit code: $LASTEXITCODE)" -ForegroundColor Red
}
