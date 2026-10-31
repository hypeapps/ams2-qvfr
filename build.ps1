# Builds the project: loads the MSVC x64 environment (vcvars64) and runs CMake + Ninja.
# Usage: .\build.ps1 [-Config Release|Debug] [-Reconfigure]
param(
    [ValidateSet('Release', 'Debug')][string]$Config = 'Release',
    [switch]$Reconfigure
)

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
    $vs = & "$installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { Write-Error 'Visual Studio with the C++ x64 tools was not found.'; exit 1 }
    # vcvars calls vswhere from PATH and prints warnings to stderr, so both streams are silenced.
    $env:PATH = "$installer;$env:PATH"
    cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}

Push-Location $PSScriptRoot
try {
    if ($Reconfigure -or -not (Test-Path build\CMakeCache.txt)) {
        cmake --preset default
        if ($LASTEXITCODE) { Write-Error "cmake configure: $LASTEXITCODE"; exit $LASTEXITCODE }
    }
    cmake --build --preset ($Config.ToLower())
    if ($LASTEXITCODE) { Write-Error "cmake build: $LASTEXITCODE"; exit $LASTEXITCODE }
    Write-Host "OK: build\bin\$Config"
} finally {
    Pop-Location
}
