param(
    [string]$Job = "job_descriptions\job_002_machine_learning_engineer.txt",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if (-not $root) {
    $root = (Get-Location).Path
}

Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

if ($Rebuild) {
    Write-Host ""
    Write-Host "== Step 1: Rebuilding the vector database ==" -ForegroundColor Cyan
    & $python ".\src\resume_rag.py"
}

Write-Host ""
Write-Host "== Step 2: Matching the job description ==" -ForegroundColor Cyan
& $python ".\src\job_matcher.py" $Job

$jobBase = [System.IO.Path]::GetFileNameWithoutExtension($Job)
$result = Join-Path $root "outputs\$jobBase-results.json"

Write-Host ""
Write-Host "== Step 3: Candidate summary ==" -ForegroundColor Cyan
& $python ".\demo\show_results.py" $result
