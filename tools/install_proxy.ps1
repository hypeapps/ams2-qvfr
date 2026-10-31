<#
.SYNOPSIS
  Installs, uninstalls or checks the openvr_api.dll proxy (ams2-qvfr) in the AMS2 folder.

.DESCRIPTION
  Install (default):
    1. Checks that the game is not running.
    2. If x64\openvr_api.dll is the original 0.9.12 (MD5), copies it to x64\bckp_qvfr\openvr_api.dll (once).
    3. Copies from build\bin\<Config>\proxy\: openvr_api.dll (proxy), openvr_api_orig.dll (Valve original) and openvr_api.pdb.
  -Uninstall: restores the original to x64\openvr_api.dll and removes the proxy files.
  -Status: only shows the current state.

  Proxy logs and settings: Documents\ams2-qvfr\ (outside the game folder).

.EXAMPLE
  .\tools\install_proxy.ps1 -Status
  .\tools\install_proxy.ps1
  .\tools\install_proxy.ps1 -Uninstall
#>
param(
    [string]$GameDir = "D:\Steam\steamapps\common\Automobilista 2",
    [ValidateSet("Release", "Debug")][string]$Config = "Release",
    [switch]$Uninstall,
    [switch]$Status
)

$ErrorActionPreference = "Stop"  # a failed copy must stop the script, not report success
$OrigMd5 = "864be5a92276ce2b902b6f3a0f850703"  # official openvr_api.dll 0.9.12 (win64), the same one the game ships
$x64 = Join-Path $GameDir "x64"
$dll = Join-Path $x64 "openvr_api.dll"
$orig = Join-Path $x64 "openvr_api_orig.dll"
$pdb = Join-Path $x64 "openvr_api.pdb"
$backupDir = Join-Path $x64 "bckp_qvfr"
$backup = Join-Path $backupDir "openvr_api.dll"
$build = Join-Path $PSScriptRoot "..\build\bin\$Config\proxy"

function Md5($path) {
    if (-not (Test-Path $path)) { return $null }
    (Get-FileHash -Algorithm MD5 $path).Hash.ToLower()
}

function Describe($path) {
    $h = Md5 $path
    if (-not $h) { return "missing" }
    $size = (Get-Item $path).Length
    if ($h -eq $OrigMd5) { return "original OpenVR 0.9.12 ($size B)" }
    # Marker = the proxy's startup log line (current and the earlier wording).
    $markers = @("ams2-qvfr openvr_api.dll proxy", "ams2-qvfr proxy openvr_api.dll")
    if (Select-String -Path $path -Pattern $markers -SimpleMatch -Quiet) {
        return "proxy ams2-qvfr ($size B, MD5 $h)"
    }
    return "OTHER FILE ($size B, MD5 $h)"
}

function Show-Status {
    "Game folder: $x64"
    "  openvr_api.dll       : " + (Describe $dll)
    "  openvr_api_orig.dll  : " + (Describe $orig)
    "  bckp_qvfr\openvr_api : " + (Describe $backup)
}

if (-not (Test-Path $x64)) { throw "Folder $x64 does not exist" }
if ($Status) { Show-Status; return }

$running = Get-Process -Name "AMS2", "AMS2AVX" -ErrorAction SilentlyContinue
if ($running) { throw "The game is running ($($running.Name -join ', ')). Close it and run the script again." }

if ($Uninstall) {
    if ((Md5 $dll) -eq $OrigMd5) {
        "x64\openvr_api.dll is already the original."
    } elseif ((Describe $dll) -notlike "proxy ams2-qvfr*") {
        throw "x64\openvr_api.dll is neither the original nor our proxy ($(Describe $dll)). Nothing changed."
    } elseif ((Md5 $orig) -eq $OrigMd5) {
        Copy-Item $orig $dll -Force
        "Restored the original from openvr_api_orig.dll."
    } elseif ((Md5 $backup) -eq $OrigMd5) {
        Copy-Item $backup $dll -Force
        "Restored the original from bckp_qvfr."
    } else {
        throw "Original 0.9.12 not found (neither openvr_api_orig.dll nor bckp_qvfr). Nothing changed."
    }
    if ((Md5 $dll) -ne $OrigMd5) { throw "After restoring, x64\openvr_api.dll is not the original." }
    foreach ($f in @($orig, $pdb)) { if (Test-Path $f) { Remove-Item $f -Force } }
    "Proxy uninstalled. The bckp_qvfr backup stays."
    Show-Status
    return
}

# --- install
$srcDll = Join-Path $build "openvr_api.dll"
$srcOrig = Join-Path $build "openvr_api_orig.dll"
$srcPdb = Join-Path $build "openvr_api.pdb"
if (-not (Test-Path $srcDll)) { throw "$srcDll not found. Run .\build.ps1 first" }
if ((Md5 $srcOrig) -ne $OrigMd5) { throw "$srcOrig is not the official 0.9.12 DLL." }

$current = Md5 $dll
if ($current -eq $OrigMd5) {
    if (-not (Test-Path $backup)) {
        New-Item -ItemType Directory -Force $backupDir | Out-Null
        Copy-Item $dll $backup
        if ((Md5 $backup) -ne $OrigMd5) { throw "Backup $backup does not match the original. Nothing else changed." }
        "Original backed up to: $backup"
    }
} elseif ((Describe $dll) -like "proxy ams2-qvfr*") {
    "Updating a previously installed proxy."
} elseif ($current) {
    throw "x64\openvr_api.dll is neither the original 0.9.12 nor our proxy (MD5 $current). OpenComposite? Nothing changed."
}

Copy-Item $srcOrig $orig -Force
Copy-Item $srcDll $dll -Force
if (Test-Path $srcPdb) { Copy-Item $srcPdb $pdb -Force }
if ((Md5 $dll) -ne (Md5 $srcDll) -or (Md5 $orig) -ne $OrigMd5) { throw "Copied files do not match the build. Check x64\ manually." }
"Installed the ams2-qvfr proxy."
Show-Status
"Logs and settings: $([Environment]::GetFolderPath('MyDocuments'))\ams2-qvfr\"
