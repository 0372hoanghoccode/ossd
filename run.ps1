Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ollamaExe = Join-Path $root "tools\ollama\ollama.exe"
$streamlitExe = Join-Path $root ".venv\Scripts\streamlit.exe"
$appFile = Join-Path $root "app.py"
$modelsDir = Join-Path $root "tools\ollama\models"
$model = "qwen2.5:7b"

if (-not (Test-Path -LiteralPath $ollamaExe)) {
    throw "Khong tim thay Ollama tai: $ollamaExe"
}
if (-not (Test-Path -LiteralPath $streamlitExe)) {
    throw "Khong tim thay Streamlit trong venv tai: $streamlitExe"
}
if (-not (Test-Path -LiteralPath $appFile)) {
    throw "Khong tim thay app.py tai: $appFile"
}

$env:OLLAMA_MODELS = $modelsDir
if (-not $env:OLLAMA_HOST) {
    $env:OLLAMA_HOST = "127.0.0.1:11434"
}
$env:OLLAMA_MODEL = $model

$serveProc = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "ollama.exe" -and
        $_.CommandLine -match [regex]::Escape($ollamaExe) -and
        $_.CommandLine -match "\sserve(\s|$)"
    } |
    Select-Object -First 1

if (-not $serveProc) {
    Write-Host "Dang khoi dong Ollama service..."
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden -WorkingDirectory (Split-Path -Parent $ollamaExe) | Out-Null
    Start-Sleep -Seconds 2
}

$ollamaList = & $ollamaExe list 2>$null | Out-String
if ($LASTEXITCODE -ne 0 -or $ollamaList -notmatch "(?m)^qwen2\.5:7b\s") {
    Write-Host "Dang tai model $model (neu chua co)..."
    & $ollamaExe pull $model
    if ($LASTEXITCODE -ne 0) {
        throw "Khong the tai model $model"
    }
}

Write-Host "Mo ung dung voi OLLAMA_MODEL=$model"
& $streamlitExe run $appFile
