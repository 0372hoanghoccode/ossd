# Encoding: UTF-8 with BOM
param(
    [string]$Pdf = "review_on_tap.pdf",
    [string]$EvalSet = "data/review_on_tap_eval_vi.json",
    [string]$QuestionSetKey = "table2_questions",
    [string[]]$QuestionIds = @("V1", "V2", "V3"),
    [int[]]$ChunkSizes = @(500, 1000, 1500, 2000),
    [int[]]$ChunkOverlaps = @(50, 100, 200),
    [string]$RetrievalMode = "hybrid",
    [string]$OllamaModel = "qwen2.5:7b",
    [int]$RetrieverK = 3,
    [string]$Output = "documentation/chunk_benchmark_multi_review_on_tap_qwen2.5_7b.md"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Khong tim thay Python venv tai: $VenvPython" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Pdf)) {
    Write-Host "Khong tim thay file PDF: $Pdf" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $EvalSet)) {
    Write-Host "Khong tim thay file eval set: $EvalSet" -ForegroundColor Red
    exit 1
}

$argsList = @(
    "tools/chunk_benchmark_multi.py",
    "--pdf", $Pdf,
    "--eval-set", $EvalSet,
    "--question-set-key", $QuestionSetKey,
    "--retrieval-mode", $RetrievalMode,
    "--ollama-model", $OllamaModel,
    "--retriever-k", $RetrieverK,
    "--output", $Output
)

if ($QuestionIds.Count -gt 0) {
    $argsList += "--question-ids"
    $argsList += $QuestionIds
}

if ($ChunkSizes.Count -gt 0) {
    $argsList += "--chunk-sizes"
    $argsList += $ChunkSizes
}

if ($ChunkOverlaps.Count -gt 0) {
    $argsList += "--chunk-overlaps"
    $argsList += $ChunkOverlaps
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  MULTI-QUESTION CHUNK BENCHMARK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Pdf:            $Pdf"
Write-Host "  EvalSet:        $EvalSet"
Write-Host "  QuestionSetKey: $QuestionSetKey"
Write-Host "  QuestionIds:    $($QuestionIds -join ', ')"
Write-Host "  ChunkSizes:     $($ChunkSizes -join ', ')"
Write-Host "  ChunkOverlaps:  $($ChunkOverlaps -join ', ')"
Write-Host "  RetrievalMode:  $RetrievalMode"
Write-Host "  OllamaModel:    $OllamaModel"
Write-Host "  Output:         $Output"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

& $VenvPython $argsList

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Benchmark hoan tat. Bao cao:" -ForegroundColor Green
    Write-Host "  $Output" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Benchmark that bai (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}
