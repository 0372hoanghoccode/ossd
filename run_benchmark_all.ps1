# =============================================================================
# run_benchmark_all.ps1 - Chạy benchmark trên NHIỀU model, mỗi model 1 file
# =============================================================================
# Cách dùng:
#   .\run_benchmark_all.ps1
#   .\run_benchmark_all.ps1 -Question "Câu hỏi của bạn"
#   .\run_benchmark_all.ps1 -Models @("qwen2.5:7b", "qwen2.5:0.5b")
# =============================================================================

param(
    [string]$Pdf = "review_on_tap.pdf",
    [string]$Question = "",
    [string[]]$References = @(),
    [string[]]$Models = @("qwen2.5:7b", "qwen2.5:0.5b"),
    [string]$RetrievalMode = "hybrid",
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

# Prompt for references if not provided
if ($References.Count -eq 0) {
    $refInput = Read-Host "Nhập câu trả lời chuẩn (reference). Bỏ trống = tự sinh"
    if ($refInput) {
        $References = @($refInput)
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK NHIEU MODEL - SO SANH CHIEN LUOC PHAN DOAN" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  PDF:      $Pdf"
Write-Host "  Question: $Question"
Write-Host "  Models:   $($Models -join ', ')"
Write-Host "  Mode:     $RetrievalMode"
if ($References.Count -gt 0) {
    Write-Host "  Ref:      $($References[0].Substring(0, [Math]::Min(60, $References[0].Length)))..."
}
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$outputFiles = @()

foreach ($model in $Models) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Yellow
    Write-Host "  RUNNING MODEL: $model" -ForegroundColor Yellow
    Write-Host "================================================================" -ForegroundColor Yellow

    $args_list = @(
        "tools/chunk_benchmark_auto.py",
        "--pdf", $Pdf,
        "--question", $Question,
        "--retrieval-mode", $RetrievalMode,
        "--ollama-model", $model,
        "--retriever-k", $RetrieverK,
        "--repeats", $Repeats
    )

    if ($References.Count -gt 0) {
        $args_list += "--references"
        $args_list += $References
    }

    & $VenvPython $args_list

    if ($LASTEXITCODE -eq 0) {
        $modelSafe = $model -replace '[^a-zA-Z0-9._-]', '_'
        $outFile = "documentation/chunk_benchmark_$modelSafe.md"
        $outputFiles += $outFile
        Write-Host "✅ Model $model hoàn tất -> $outFile" -ForegroundColor Green
    } else {
        Write-Host "❌ Model $model thất bại (exit code: $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  KET QUA" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Các file báo cáo đã tạo:" -ForegroundColor Green
foreach ($f in $outputFiles) {
    Write-Host "  📄 $f" -ForegroundColor Green
}
Write-Host ""
Write-Host "So sánh các file trên để thấy sự khác biệt giữa các model." -ForegroundColor Yellow
