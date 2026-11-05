<#
.SYNOPSIS
  Builds the release zip of the mod from a finished build (run .\build.ps1 first).

.DESCRIPTION
  The zip has the layout of the game folder, so players extract it into the folder that holds AMS2.exe:
    x64\openvr_api.dll                          the mod (proxy)
    x64\qvfr\openvr\openvr_api.dll              the official Valve 0.9.12 DLL ([runtime] path=openvr)
    x64\qvfr\opencomposite\openvr_api.dll       the pinned OpenComposite build ([runtime] path=openxr), with its
                                                licenses and SOURCE.txt
    x64\qvfr\XR_APILAYER_AMS2QVFR_eye_gaze.json OpenXR layer manifest
    x64\qvfr\qvfr_config.exe                    settings GUI
    x64\qvfr\proxy\openvr_api.dll               spare copy of the proxy (the GUI's Repair)
    x64\qvfr\nvngx_dlss.dll                     NVIDIA DLSS runtime for [dlaa] mode=dlaa (NVIDIA RTX SDKs license in
                                                licenses\nvidia-rtx-sdks-LICENSE.txt)
    x64\qvfr\README.md, x64\qvfr\licenses\      README, the mod's license and the third-party licenses
  No qvfr.ini: the proxy and the GUI create the default one, so an update keeps the player's settings.
  Every binary is checked (the Valve and OpenComposite DLLs by MD5, the DLSS DLL by SHA256, the spare proxy against the
  proxy). The build must have the DLSS SDK (tools\fetch_dlss_sdk.ps1 before .\build.ps1), so that the proxy has DLAA.
  Entries use forward slashes (Compress-Archive in Windows PowerShell 5.1 writes backslashes).

.EXAMPLE
  .\tools\package.ps1                 dist\ams2-qvfr-<version>.zip from build\bin\Release
#>
param(
    [string]$Config = "Release",
    [string]$OutDir = ""
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem

$root = Split-Path $PSScriptRoot -Parent
$build = Join-Path $root "build\bin\$Config\proxy"
if (-not $OutDir) { $OutDir = Join-Path $root "dist" }
$ValveMd5 = "864be5a92276ce2b902b6f3a0f850703"
$OpenCompositeMd5 = "1a89f94450ab870d74b5c20da2e52d85"
$OpenCompositeDir = Join-Path $root "third_party\opencomposite-1.0.1539"
$OpenCompositeCommit = "a27e7e6a64bdcd1eff6b7fba1ea2ea34bcf1273d"
$DlssSdk = Join-Path $root "third_party\dlss-310.9.16"
$DlssSha256 = "d00b5b0ab29c3ca101b70c3a5ecb3a01b7ef33766a3383a28a18623083e6123e"  # nvngx_dlss.dll 310.9.16 (fetch_dlss_sdk.ps1)

function Md5($p) { (Get-FileHash $p -Algorithm MD5).Hash.ToLower() }
function Need($p) { if (-not (Test-Path $p)) { throw "$p not found. Run .\build.ps1 first." }; $p }

$version = ([regex]::Match((Get-Content (Join-Path $root "CMakeLists.txt") -Raw), "project\(\w+ VERSION ([0-9.]+)")).Groups[1].Value
if (-not $version) { throw "No project version in CMakeLists.txt." }

$proxy = Need (Join-Path $build "openvr_api.dll")
$valve = Need (Join-Path $build "qvfr\openvr\openvr_api.dll")
$oc = Need (Join-Path $build "qvfr\opencomposite\openvr_api.dll")
$manifest = Need (Join-Path $build "qvfr\XR_APILAYER_AMS2QVFR_eye_gaze.json")
$gui = Need (Join-Path $build "qvfr\qvfr_config.exe")
$spare = Need (Join-Path $build "qvfr\proxy\openvr_api.dll")
$dlss = Join-Path $build "qvfr\nvngx_dlss.dll"
if (-not (Test-Path $dlss)) {
    throw "$dlss not found: the build has no DLSS SDK, so the proxy has no DLAA. Run .\tools\fetch_dlss_sdk.ps1, then .\build.ps1."
}
if ((Get-FileHash $dlss -Algorithm SHA256).Hash.ToLower() -ne $DlssSha256) { throw "$dlss is not the pinned NVIDIA DLSS 310.9.16 DLL." }
if ((Md5 $valve) -ne $ValveMd5) { throw "$valve is not the official Valve 0.9.12 DLL." }
if ((Md5 $oc) -ne $OpenCompositeMd5) { throw "$oc is not the pinned OpenComposite build (MD5 $OpenCompositeMd5)." }
if ((Md5 $spare) -ne (Md5 $proxy)) { throw "$spare is not the same build as $proxy. Run .\build.ps1 first." }

$sourceTxt = @"
OpenComposite 1.0.1539, the official AppVeyor build of commit $OpenCompositeCommit, unmodified.
License: GNU General Public License v3 (LICENSE.txt), parts under the licenses in the other LICENSE files here.
Complete corresponding source code: https://gitlab.com/znixian/OpenOVR/-/tree/$OpenCompositeCommit
Project: https://gitlab.com/znixian/OpenOVR
"@

# Published path in the zip -> source file (or text, for generated files).
$files = [ordered]@{
    "x64/openvr_api.dll"                           = $proxy
    "x64/qvfr/openvr/openvr_api.dll"               = $valve
    "x64/qvfr/opencomposite/openvr_api.dll"        = $oc
    "x64/qvfr/XR_APILAYER_AMS2QVFR_eye_gaze.json"  = $manifest
    "x64/qvfr/qvfr_config.exe"                     = $gui
    "x64/qvfr/proxy/openvr_api.dll"                = $spare
    "x64/qvfr/nvngx_dlss.dll"                      = $dlss
    "x64/qvfr/README.md"                           = Need (Join-Path $root "README.md")
    "x64/qvfr/licenses/LICENSE.md"                 = Need (Join-Path $root "LICENSE.md")
    "x64/qvfr/licenses/THIRD_PARTY.md"             = Need (Join-Path $root "THIRD_PARTY.md")
    "x64/qvfr/licenses/minhook-LICENSE.txt"        = Need (Join-Path $root "third_party\minhook\LICENSE.txt")
    "x64/qvfr/licenses/openvr-LICENSE.txt"         = Need (Join-Path $root "third_party\openvr-0.9.12\LICENSE")
    "x64/qvfr/licenses/openxr-sdk-LICENSE.txt"     = Need (Join-Path $root "third_party\openxr-1.1.65\LICENSE")
    "x64/qvfr/licenses/rmlui-LICENSE.txt"          = Need (Join-Path $root "third_party\rmlui-6.3\LICENSE.txt")
    "x64/qvfr/licenses/freetype-FTL.txt"           = Need (Join-Path $root "third_party\freetype-2.15.0\docs\FTL.TXT")
    "x64/qvfr/licenses/inter-OFL.txt"              = Need (Join-Path $root "third_party\fonts\inter\LICENSE.txt")
    "x64/qvfr/licenses/barlowcondensed-OFL.txt"    = Need (Join-Path $root "third_party\fonts\barlowcondensed\OFL.txt")
    "x64/qvfr/licenses/tabler-icons-LICENSE.txt"   = Need (Join-Path $root "third_party\fonts\tabler-icons\LICENSE")
    "x64/qvfr/licenses/nvidia-rtx-sdks-LICENSE.txt" = Need (Join-Path $DlssSdk "LICENSE.txt")
}
foreach ($f in Get-ChildItem $OpenCompositeDir -File) { $files["x64/qvfr/opencomposite/$($f.Name)"] = $f.FullName }

New-Item -ItemType Directory -Force $OutDir | Out-Null
$zipPath = Join-Path (Resolve-Path $OutDir) "ams2-qvfr-$version.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($e in $files.GetEnumerator()) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $e.Value, $e.Key,
            [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
    $entry = $zip.CreateEntry("x64/qvfr/opencomposite/SOURCE.txt")
    $w = New-Object System.IO.StreamWriter($entry.Open(), (New-Object System.Text.UTF8Encoding $false))
    $w.Write($sourceTxt.Replace("`r`n", "`n").Replace("`n", "`r`n") + "`r`n")
    $w.Dispose()
} finally {
    $zip.Dispose()
}

"Package: $zipPath ($((Get-Item $zipPath).Length) B, version $version)"
"  proxy MD5 $(Md5 $proxy), GUI MD5 $(Md5 $gui), OpenComposite MD5 $(Md5 $oc)"
$check = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try { $check.Entries | ForEach-Object { "  {0,10}  {1}" -f $_.Length, $_.FullName } } finally { $check.Dispose() }
