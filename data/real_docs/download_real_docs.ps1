$ErrorActionPreference = "Stop"

$outputDir = Join-Path $PSScriptRoot "downloads"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$sources = @(
    @{ Name = "nist_sp_800_53r5.pdf"; Url = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf" },
    @{ Name = "nist_sp_800_61r2.pdf"; Url = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf" },
    @{ Name = "nist_sp_800_40r4.pdf"; Url = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-40r4.pdf" }
)

$results = @()

foreach ($src in $sources) {
    $outFile = Join-Path $outputDir $src.Name
    try {
        Invoke-WebRequest -Uri $src.Url -OutFile $outFile -UseBasicParsing
        $size = (Get-Item $outFile).Length
        $results += [PSCustomObject]@{
            file = $src.Name
            status = "ok"
            bytes = $size
            url = $src.Url
        }
    }
    catch {
        $results += [PSCustomObject]@{
            file = $src.Name
            status = "failed"
            bytes = 0
            url = $src.Url
            error = $_.Exception.Message
        }
    }
}

$reportPath = Join-Path $outputDir "download_report.json"
$results | ConvertTo-Json -Depth 6 | Set-Content -Path $reportPath -Encoding UTF8

Write-Output "Saved files to: $outputDir"
Write-Output "Report: $reportPath"
$results | Format-Table -AutoSize | Out-String | Write-Output
