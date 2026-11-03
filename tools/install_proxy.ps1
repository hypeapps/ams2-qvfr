<#
.SYNOPSIS
  Installs, uninstalls or checks the ams2-qvfr mod (openvr_api.dll proxy) in the AMS2 folder.

.DESCRIPTION
  Layout in the game's x64\ folder (docs/architecture.md):
    openvr_api.dll (+ .pdb)                     the proxy
    qvfr\openvr\openvr_api.dll                  the official Valve 0.9.12 DLL ([runtime] path=openvr)
    qvfr\opencomposite\openvr_api.dll           OpenComposite ([runtime] path=openxr)
    qvfr\XR_APILAYER_AMS2QVFR_eye_gaze.json     OpenXR layer manifest (points at ..\openvr_api.dll)
    qvfr\qvfr_config.exe                        settings GUI
    qvfr\proxy\openvr_api.dll                   spare copy of the proxy (the GUI's Repair puts it back in x64\)
    qvfr\nvngx_dlss.dll                         NVIDIA DLSS ([dlaa] mode=dlaa), only from a build with the local DLSS SDK
    qvfr\qvfr.ini, qvfr\logs\, bench\, shots\   settings and output (created by the proxy)

  Install (default):
    1. Checks that the game is not running.
    2. The file that was x64\openvr_api.dll before the mod (Valve or OpenComposite) is copied once to
       x64\bckp_qvfr\openvr_api.dll; uninstall puts it back. OpenComposite found there also becomes the
       OpenXR-path backend unless -OpenComposite names another DLL.
    3. Copies the build (build\bin\<Config>\proxy\) in the layout above and removes the pre-M8.2 file
       x64\openvr_api_orig.dll.
  -OpenComposite <dll>: the OpenComposite DLL for qvfr\opencomposite\.
  -Path openvr|openxr: sets [runtime] path in x64\qvfr\qvfr.ini (takes effect at the next game start).
  -Uninstall: restores x64\openvr_api.dll from the backup and removes the mod's binaries (also the GUI and the
    spare proxy); qvfr.ini and the output folders in x64\qvfr\ stay.
  -Status: only shows the current state.

.EXAMPLE
  .\tools\install_proxy.ps1 -Status
  .\tools\install_proxy.ps1 -Path openvr
  .\tools\install_proxy.ps1 -OpenComposite "D:\Steam\steamapps\common\Automobilista 2\x64\bckp_opencomposite_openvrddl\openvr_api.dll" -Path openxr
  .\tools\install_proxy.ps1 -Uninstall
#>
param(
    [string]$GameDir = "D:\Steam\steamapps\common\Automobilista 2",
    [ValidateSet("Release", "Debug")][string]$Config = "Release",
    [string]$OpenComposite = "",
    [ValidateSet("", "openvr", "openxr")][string]$Path = "",
    [switch]$Uninstall,
    [switch]$Status
)

$ErrorActionPreference = "Stop"  # a failed copy must stop the script, not report success
$ValveMd5 = "864be5a92276ce2b902b6f3a0f850703"  # official openvr_api.dll 0.9.12 (win64), the same one the game ships
$LayerManifest = "XR_APILAYER_AMS2QVFR_eye_gaze.json"
$x64 = Join-Path $GameDir "x64"
$dll = Join-Path $x64 "openvr_api.dll"
$pdb = Join-Path $x64 "openvr_api.pdb"
$oldOrig = Join-Path $x64 "openvr_api_orig.dll"  # pre-M8.2 layout
$qvfr = Join-Path $x64 "qvfr"
$valve = Join-Path $qvfr "openvr\openvr_api.dll"
$ocDll = Join-Path $qvfr "opencomposite\openvr_api.dll"
$manifest = Join-Path $qvfr $LayerManifest
$gui = Join-Path $qvfr "qvfr_config.exe"
$spare = Join-Path $qvfr "proxy\openvr_api.dll"
$dlss = Join-Path $qvfr "nvngx_dlss.dll"
$ini = Join-Path $qvfr "qvfr.ini"
$backupDir = Join-Path $x64 "bckp_qvfr"
$backup = Join-Path $backupDir "openvr_api.dll"
$build = Join-Path $PSScriptRoot "..\build\bin\$Config\proxy"

function Md5($p) {
    if (-not (Test-Path $p)) { return $null }
    (Get-FileHash -Algorithm MD5 $p).Hash.ToLower()
}

function IsProxy($p) {
    # Marker = the proxy's startup log line (current and the earlier wording).
    (Test-Path $p) -and (Select-String -Path $p -Pattern @("ams2-qvfr openvr_api.dll proxy", "ams2-qvfr proxy openvr_api.dll") -SimpleMatch -Quiet)
}

function IsOpenComposite($p) {
    (Test-Path $p) -and -not (IsProxy $p) -and (Select-String -Path $p -Pattern "OpenComposite" -SimpleMatch -Quiet)
}

function Describe($p) {
    $h = Md5 $p
    if (-not $h) { return "missing" }
    $size = (Get-Item $p).Length
    if ($h -eq $ValveMd5) { return "Valve OpenVR 0.9.12 ($size B)" }
    if (IsProxy $p) { return "proxy ams2-qvfr ($size B, MD5 $h)" }
    if (IsOpenComposite $p) { return "OpenComposite ($size B, MD5 $h)" }
    return "OTHER FILE ($size B, MD5 $h)"
}

function IniPath() {
    if (-not (Test-Path $ini)) { return "(no qvfr.ini yet: path=openxr by default)" }
    $line = Select-String -Path $ini -Pattern "^\s*path\s*=" | Select-Object -First 1
    if ($line) { return $line.Line.Trim() } else { return "(no path key: path=openxr)" }
}

function Show-Status {
    "Game folder: $x64"
    "  openvr_api.dll                 : " + (Describe $dll)
    "  qvfr\openvr\openvr_api.dll     : " + (Describe $valve)
    "  qvfr\opencomposite\openvr_api  : " + (Describe $ocDll)
    "  qvfr\$LayerManifest : " + $(if (Test-Path $manifest) { "present" } else { "missing" })
    "  qvfr\qvfr_config.exe           : " + $(if (Test-Path $gui) { "present (MD5 $(Md5 $gui))" } else { "missing" })
    "  qvfr\nvngx_dlss.dll            : " + $(if (Test-Path $dlss) { "present ($((Get-Item $dlss).Length) B)" } else { "missing" })
    "  qvfr\proxy\openvr_api.dll      : " + $(if (-not (Test-Path $spare)) { "missing" } elseif ((Md5 $spare) -eq (Md5 $dll)) { "same as x64\openvr_api.dll" } else { Describe $spare })
    "  qvfr\qvfr.ini [runtime]        : " + (IniPath)
    "  bckp_qvfr\openvr_api.dll       : " + (Describe $backup)
    if (Test-Path $oldOrig) { "  openvr_api_orig.dll (old layout): " + (Describe $oldOrig) }
}

# Sets [runtime] path=<value> in qvfr.ini, keeping everything else (a new file gets only this section; the proxy
# uses the defaults for the rest).
function Set-IniPath($value) {
    New-Item -ItemType Directory -Force $qvfr | Out-Null
    if (-not (Test-Path $ini)) {
        Set-Content -Path $ini -Value "[runtime]`r`npath=$value" -Encoding ascii
        return "Created $ini with [runtime] path=$value (the proxy uses its defaults for the other settings)."
    }
    $lines = [System.Collections.Generic.List[string]](Get-Content $ini)
    $sec = -1; $key = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s*\[(.+)\]\s*$') {
            if ($sec -ge 0 -and $key -lt 0) { break }
            $sec = if ($matches[1] -eq "runtime") { $i } else { -1 }
        } elseif ($sec -ge 0 -and $lines[$i] -match '^\s*path\s*=') { $key = $i; break }
    }
    if ($key -ge 0) { $lines[$key] = "path=$value" }
    elseif ($sec -ge 0) { $lines.Insert($sec + 1, "path=$value") }
    else { $lines.Insert(0, ""); $lines.Insert(0, "path=$value"); $lines.Insert(0, "[runtime]") }
    # ASCII like the proxy's own file: GetPrivateProfileString would read a UTF-8 BOM as part of the first line.
    Set-Content -Path $ini -Value $lines -Encoding ascii
    return "qvfr.ini: [runtime] path=$value"
}

if (-not (Test-Path $x64)) { throw "Folder $x64 does not exist" }
if ($Status) { Show-Status; return }

$running = Get-Process -Name "AMS2", "AMS2AVX" -ErrorAction SilentlyContinue
if ($running) { throw "The game is running ($($running.Name -join ', ')). Close it and run the script again." }

if ($Uninstall) {
    if (-not (IsProxy $dll)) {
        "x64\openvr_api.dll is not our proxy ($(Describe $dll)); only leftovers of the mod are removed."
    } else {
        $restore = if (Test-Path $backup) { $backup } elseif ((Md5 $valve) -eq $ValveMd5) { $valve } elseif ((Md5 $oldOrig) -eq $ValveMd5) { $oldOrig } else { $null }
        if (-not $restore) { throw "No backup of the pre-mod openvr_api.dll (bckp_qvfr) and no Valve DLL. Nothing changed." }
        Copy-Item $restore $dll -Force
        if ((Md5 $dll) -ne (Md5 $restore)) { throw "After restoring, x64\openvr_api.dll does not match $restore." }
        "Restored x64\openvr_api.dll from $restore ($(Describe $dll))."
    }
    foreach ($f in @($pdb, $oldOrig, $valve, $ocDll, $manifest, $gui, $spare, $dlss)) { if (Test-Path $f) { Remove-Item $f -Force } }
    foreach ($d in @((Join-Path $qvfr "openvr"), (Join-Path $qvfr "opencomposite"), (Join-Path $qvfr "proxy"))) {
        if ((Test-Path $d) -and -not (Get-ChildItem $d -Force)) { Remove-Item $d -Force }
    }
    "Mod binaries removed. Kept: bckp_qvfr\ and x64\qvfr\ (qvfr.ini, logs, bench, shots), if present."
    Show-Status
    return
}

# --- install
$srcDll = Join-Path $build "openvr_api.dll"
$srcPdb = Join-Path $build "openvr_api.pdb"
$srcValve = Join-Path $build "qvfr\openvr\openvr_api.dll"
$srcManifest = Join-Path $build "qvfr\$LayerManifest"
$srcGui = Join-Path $build "qvfr\qvfr_config.exe"
$srcSpare = Join-Path $build "qvfr\proxy\openvr_api.dll"
$srcDlss = Join-Path $build "qvfr\nvngx_dlss.dll"
if (-not (Test-Path $srcDll)) { throw "$srcDll not found. Run .\build.ps1 first" }
if ((Md5 $srcValve) -ne $ValveMd5) { throw "$srcValve is not the official 0.9.12 DLL." }
if (-not (Test-Path $srcManifest)) { throw "$srcManifest not found. Run .\build.ps1 first" }
if (-not (Test-Path $srcGui)) { throw "$srcGui not found. Run .\build.ps1 first" }
if ((Md5 $srcSpare) -ne (Md5 $srcDll)) { throw "$srcSpare is not the same build as $srcDll. Run .\build.ps1 first" }
if ($OpenComposite -and -not (IsOpenComposite $OpenComposite)) {
    throw "$OpenComposite is not OpenComposite ($(Describe $OpenComposite)). Nothing changed."
}

# The pre-mod file: backed up once; OpenComposite there becomes the OpenXR backend.
if (IsProxy $dll) {
    "Updating a previously installed proxy."
} elseif (Test-Path $dll) {
    if (-not (Test-Path $backup)) {
        New-Item -ItemType Directory -Force $backupDir | Out-Null
        Copy-Item $dll $backup
        if ((Md5 $backup) -ne (Md5 $dll)) { throw "Backup $backup does not match x64\openvr_api.dll. Nothing else changed." }
        "Pre-mod x64\openvr_api.dll ($(Describe $dll)) backed up to $backup"
    } elseif ((Md5 $backup) -ne (Md5 $dll)) {
        throw "x64\openvr_api.dll ($(Describe $dll)) differs from the existing backup ($(Describe $backup)). Move one of them away first. Nothing changed."
    }
    if (-not $OpenComposite -and (IsOpenComposite $dll)) { $OpenComposite = $backup }
}

Copy-Item $srcDll $dll -Force
if (Test-Path $srcPdb) { Copy-Item $srcPdb $pdb -Force }
New-Item -ItemType Directory -Force (Split-Path $valve) | Out-Null
Copy-Item $srcValve $valve -Force
Copy-Item $srcManifest $manifest -Force
Copy-Item $srcGui $gui -Force
New-Item -ItemType Directory -Force (Split-Path $spare) | Out-Null
Copy-Item $srcSpare $spare -Force
if (Test-Path $srcDlss) {
    Copy-Item $srcDlss $dlss -Force
    if ((Md5 $dlss) -ne (Md5 $srcDlss)) { throw "The nvngx_dlss.dll copy does not match the build." }
    "NVIDIA DLSS copied to $dlss ([dlaa] mode=dlaa)."
}
if ($OpenComposite) {
    New-Item -ItemType Directory -Force (Split-Path $ocDll) | Out-Null
    Copy-Item $OpenComposite $ocDll -Force
    if ((Md5 $ocDll) -ne (Md5 $OpenComposite)) { throw "The OpenComposite copy does not match $OpenComposite." }
    "OpenComposite copied to $ocDll (backend of [runtime] path=openxr)."
}
if ((Md5 $dll) -ne (Md5 $srcDll) -or (Md5 $spare) -ne (Md5 $srcDll) -or (Md5 $gui) -ne (Md5 $srcGui) -or (Md5 $valve) -ne $ValveMd5) { throw "Copied files do not match the build. Check x64\ manually." }
if ((Md5 $oldOrig) -eq $ValveMd5) {
    Remove-Item $oldOrig -Force
    "Removed x64\openvr_api_orig.dll (pre-M8.2 layout; the Valve DLL is now in qvfr\openvr\)."
}
if ($Path) {
    if ($Path -eq "openxr" -and -not (Test-Path $ocDll)) { throw "-Path openxr needs OpenComposite in $ocDll (use -OpenComposite)." }
    Set-IniPath $Path
}
"Installed the ams2-qvfr proxy."
Show-Status
