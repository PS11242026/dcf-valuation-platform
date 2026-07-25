#requires -Version 7
<#
.SYNOPSIS
    End-to-end product showcase for the DCF platform.

.DESCRIPTION
    Generates the full deliverable set for one or more tickers:
      * Fundamentals Fade DCF (flagship, engine v3)  -> reports/valuation/<T>/fade_dcf.xlsx
      * Canonical v1 DCF (frozen reproducibility)     -> reports/valuation/<T>/canonical_dcf.xlsx
      * Triangulation (generic / v1 / fade)           -> reports/valuation/<T>/comparison/comparison.xlsx
      * PDF + PPTX exports of each (incl. the nested comparison target)
    then the corrected research manuscript (reports/paper_v2/) and -- unless -SkipLive --
    a refresh of the live paper portfolio dashboard (conf_top10) and its PDF.

    Defaults: KO (a consumer staple, the family where the discount signal conditionally
    survived correction) and NVDA (a growth contrast that exercises the reinvestment binds
    and FCF-margin tripwire). `manage update` only refreshes prices / the dashboard; it does
    NOT rebalance (only `manage rebalance` mutates the live book).

.PARAMETER Tickers
    Tickers to showcase. Default: KO, NVDA.

.PARAMETER SkipLive
    Skip the live conf_top10 dashboard refresh + export.

.PARAMETER DryRun
    Print each command without executing it (validates the pipeline offline).

.EXAMPLE
    pwsh scripts/build_showcase.ps1
.EXAMPLE
    pwsh scripts/build_showcase.ps1 -Tickers KO,PG -SkipLive
#>
[CmdletBinding()]
param(
    [string[]] $Tickers = @('KO', 'NVDA'),
    [switch]   $SkipLive,
    [switch]   $DryRun
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot      # repo root (this script lives in scripts/)
Set-Location $root
$py = 'python'
$failures = @()

# Normalize tickers: `pwsh -File ... -Tickers KO,NVDA` passes one literal "KO,NVDA"
# string, so split on commas (also handles the default array, which has no commas).
$Tickers = $Tickers | ForEach-Object { $_ -split ',' } | Where-Object { $_ } | ForEach-Object { $_.Trim().ToUpper() }

function Invoke-Step {
    param([string] $Description, [string[]] $Cmd)
    Write-Host ""
    Write-Host "==> $Description" -ForegroundColor Cyan
    Write-Host "    $($Cmd -join ' ')" -ForegroundColor DarkGray
    if ($DryRun) { return }
    & $Cmd[0] @($Cmd[1..($Cmd.Count - 1)])
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    ! exit $LASTEXITCODE" -ForegroundColor Yellow
        $script:failures += $Description
    }
}

Write-Host "DCF platform showcase  (root: $root)" -ForegroundColor Green
Write-Host "Tickers: $($Tickers -join ', ')   SkipLive: $SkipLive   DryRun: $DryRun"

foreach ($t in $Tickers) {
    Invoke-Step "$t  --  Fundamentals Fade DCF (flagship, v3)" @($py, '-m', 'dcf_project', 'valuation', 'dcf', $t)
    Invoke-Step "$t  --  Canonical v1 DCF (frozen)"            @($py, '-m', 'dcf_project', 'valuation', 'canonical', $t)
    Invoke-Step "$t  --  Triangulation (generic/v1/fade)"      @($py, '-m', 'dcf_project', 'valuation', 'compare', $t)
    Invoke-Step "$t  --  export PDF"                            @($py, '-m', 'dcf_project', 'export', 'pdf', $t)
    Invoke-Step "$t  --  export PPTX"                           @($py, '-m', 'dcf_project', 'export', 'ppt', $t)
    Invoke-Step "$t/comparison  --  export PDF"                 @($py, '-m', 'dcf_project', 'export', 'pdf', "$t/comparison")
}

Invoke-Step "Corrected research manuscript (v2)" @($py, '-m', 'dcf_project', 'research', 'build-paper')

if (-not $SkipLive) {
    Invoke-Step "Live dashboard refresh (conf_top10; no rebalance)" `
        @($py, '-m', 'dcf_project', 'manage', 'update', '--book', 'conf_top10')
    Invoke-Step "conf_top10  --  export PDF" @($py, '-m', 'dcf_project', 'export', 'pdf', 'conf_top10')
}

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete (no commands executed)." -ForegroundColor Green
}
elseif ($failures.Count -eq 0) {
    Write-Host "Showcase complete. Artifacts under reports/valuation/<T>/, reports/paper_v2/, reports/managed_portfolios/conf_top10/." -ForegroundColor Green
}
else {
    Write-Host "Showcase finished with $($failures.Count) failed step(s):" -ForegroundColor Yellow
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    exit 1
}
