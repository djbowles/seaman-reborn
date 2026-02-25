# Ralph Wiggum - Long-running AI agent loop for Seaman Reborn
# Usage: ./ralph.ps1 [-tool amp|claude] [-maxIterations 51]

param(
    [string]$tool = "claude",
    [int]$maxIterations = 10
)

$ErrorActionPreference = "Stop"

# Validate tool choice
if ($tool -ne "amp" -and $tool -ne "claude") {
    Write-Error "Invalid tool '$tool'. Must be 'amp' or 'claude'."
    exit 1
}

$ScriptDir = $PSScriptRoot
$PrdFile = Join-Path $ScriptDir "prd.json"
$ProgressFile = Join-Path $ScriptDir "progress.txt"
$ArchiveDir = Join-Path $ScriptDir "archive"
$LastBranchFile = Join-Path $ScriptDir ".last-branch"

# Archive previous run if branch changed
if ((Test-Path $PrdFile) -and (Test-Path $LastBranchFile)) {
    try {
        $CurrentBranch = python -c "import json; print(json.load(open(r'$PrdFile')).get('branchName',''))" 2>$null
        $LastBranch = Get-Content $LastBranchFile -ErrorAction SilentlyContinue

        if ($CurrentBranch -and $LastBranch -and ($CurrentBranch -ne $LastBranch)) {
            $Date = Get-Date -Format "yyyy-MM-dd"
            $FolderName = $LastBranch -replace '^ralph/', ''
            $ArchiveFolder = Join-Path $ArchiveDir "$Date-$FolderName"

            Write-Host "Archiving previous run: $LastBranch"
            New-Item -ItemType Directory -Path $ArchiveFolder -Force | Out-Null
            if (Test-Path $PrdFile) { Copy-Item $PrdFile $ArchiveFolder }
            if (Test-Path $ProgressFile) { Copy-Item $ProgressFile $ArchiveFolder }
            Write-Host "   Archived to: $ArchiveFolder"

            @("# Ralph Progress Log", "Started: $(Get-Date)", "---") | Set-Content $ProgressFile
        }
    } catch {
        # Ignore archive errors
    }
}

# Track current branch
if (Test-Path $PrdFile) {
    try {
        $CurrentBranch = python -c "import json; print(json.load(open(r'$PrdFile')).get('branchName',''))" 2>$null
        if ($CurrentBranch) {
            $CurrentBranch | Set-Content $LastBranchFile -NoNewline
        }
    } catch {
        # Ignore
    }
}

# Initialize progress file if it doesn't exist
if (-not (Test-Path $ProgressFile)) {
    @("# Ralph Progress Log", "Started: $(Get-Date)", "---") | Set-Content $ProgressFile
}

Write-Host "Starting Ralph - Tool: $tool - Max iterations: $maxIterations"

for ($i = 1; $i -le $maxIterations; $i++) {
    Write-Host ""
    Write-Host "==============================================================="
    Write-Host "  Ralph Iteration $i of $maxIterations ($tool)"
    Write-Host "==============================================================="

    $tempOutput = Join-Path $env:TEMP "ralph-output-$i.txt"
    $exitCode = 0

    try {
        if ($tool -eq "amp") {
            $promptFile = Join-Path $ScriptDir "prompt.md"
            Get-Content $promptFile -Raw | amp --dangerously-allow-all 2>&1 | Tee-Object -FilePath $tempOutput
        } else {
            $claudeFile = Join-Path $ScriptDir "CLAUDE.md"
            Get-Content $claudeFile -Raw | claude --dangerously-skip-permissions --print 2>&1 | Tee-Object -FilePath $tempOutput
        }
        $exitCode = $LASTEXITCODE
    } catch {
        Write-Host "Iteration $i encountered an error: $_"
    }

    # Check for completion signal
    $output = Get-Content $tempOutput -Raw -ErrorAction SilentlyContinue
    Remove-Item $tempOutput -ErrorAction SilentlyContinue

    if ($output -and $output -match "<promise>COMPLETE</promise>") {
        Write-Host ""
        Write-Host "Ralph completed all tasks!"
        Write-Host "Completed at iteration $i of $maxIterations"
        exit 0
    }

    Write-Host "Iteration $i complete. Continuing..."
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "Ralph reached max iterations ($maxIterations) without completing all tasks."
Write-Host "Check $ProgressFile for status."
exit 1
