# NexQ v2.26.0 NeMo Parakeet TDT patch, revision 3 - Windows launcher.
# The installer itself is apply-patch.mjs (Node.js is already required by the NexQ project).
#   .\apply-patch.ps1 -ProjectRoot C:\path\to\NexQ [-SkipOrtBump] [-DryRun]
param(
    [string]$ProjectRoot = ".",
    [switch]$SkipOrtBump,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw 'Node.js was not found in PATH (NexQ needs Node 20 anyway). Install it or apply ort_streaming.integration.patch by hand, see README.md.'
}
$nodeArgs = @((Join-Path $PSScriptRoot 'apply-patch.mjs'), '--project-root', $ProjectRoot)
if ($SkipOrtBump) { $nodeArgs += '--skip-ort-bump' }
if ($DryRun) { $nodeArgs += '--dry-run' }
& node @nodeArgs
exit $LASTEXITCODE
