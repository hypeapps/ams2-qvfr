<#
.SYNOPSIS
  Downloads the parts of the NVIDIA DLSS SDK that the proxy's DLAA needs into third_party\dlss-310.9.16\.

.DESCRIPTION
  The SDK stays out of git (its license allows distribution only as object code inside an application, see
  third_party/README.md), so CI fetches it before the build. Source: the public NVIDIA repository
  https://github.com/NVIDIA/DLSS at tag v310.9.16. Every file is checked against a pinned SHA256; a file that is already
  there with the right hash is not downloaded again. With the SDK in place CMake builds DLAA into the proxy, and the
  build and tools/package.ps1 take bin\nvngx_dlss.dll from there.

  Layout (ours <- NVIDIA's):
    include\*.h               <- include\*.h (the headers DLAA includes)
    lib\nvsdk_ngx_s.lib       <- lib\Windows_x86_64\x64\nvsdk_ngx_s.lib       (static NGX, Release)
    lib\nvsdk_ngx_s_dbg.lib   <- lib\Windows_x86_64\x64\nvsdk_ngx_s_dbg.lib   (static NGX, Debug)
    bin\nvngx_dlss.dll        <- lib\Windows_x86_64\rel\nvngx_dlss.dll        (DLSS runtime, Release)
    LICENSE.txt               <- LICENSE.txt                                  (NVIDIA RTX SDKs license)

.EXAMPLE
  .\tools\fetch_dlss_sdk.ps1
  .\tools\fetch_dlss_sdk.ps1 -Dest $env:RUNNER_TEMP\dlss   # elsewhere (for checking the pinned hashes)
#>
param([string]$Dest)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Dest) { $Dest = Join-Path $root "third_party\dlss-310.9.16" }
$Tag = "v310.9.16"
$Base = "https://raw.githubusercontent.com/NVIDIA/DLSS/$Tag"

# local path, path in the NVIDIA repository, SHA256
$Files = @(
    @("LICENSE.txt",                           "LICENSE.txt",                               "d4216e39ebef5f9b50a6712ebb37beeb5379862a67733a9999c651f21592aaf0"),
    @("bin/nvngx_dlss.dll",                    "lib/Windows_x86_64/rel/nvngx_dlss.dll",     "d00b5b0ab29c3ca101b70c3a5ecb3a01b7ef33766a3383a28a18623083e6123e"),
    @("lib/nvsdk_ngx_s.lib",                   "lib/Windows_x86_64/x64/nvsdk_ngx_s.lib",    "7b9a5239c1145af5f645ab472aa066f9ea79db36ac819562312aae69d65f3765"),
    @("lib/nvsdk_ngx_s_dbg.lib",               "lib/Windows_x86_64/x64/nvsdk_ngx_s_dbg.lib","d4beb60cfd83468ea1f58ac885c3e826011a1199e358ba159e7fa0604ace290e"),
    @("include/nvsdk_ngx.h",                   "include/nvsdk_ngx.h",                       "dc38e7467cf415379c9d12ae1b6e4a494c453ed92720fb53e92aecb523e7b848"),
    @("include/nvsdk_ngx_defs.h",              "include/nvsdk_ngx_defs.h",                  "ea23f33497cd274860d1c25a97644fce807dcb0037c594547203343103fad03e"),
    @("include/nvsdk_ngx_defs_dlssd.h",        "include/nvsdk_ngx_defs_dlssd.h",            "2e97fe1595b1334f71b51724f2cb67496e8c296a7a2eec5606da53055882e716"),
    @("include/nvsdk_ngx_helpers.h",           "include/nvsdk_ngx_helpers.h",               "5bcbadfe7478b802cf6d3aca4dc5ddd7d0889b99726e69c63f9e9bd555f44471"),
    @("include/nvsdk_ngx_helpers_cuda.h",      "include/nvsdk_ngx_helpers_cuda.h",          "f2851a76107bdf4fbcb7b261f3c35573c312d8ecb4bc18fb02d9e2cec7705042"),
    @("include/nvsdk_ngx_helpers_d3d.h",       "include/nvsdk_ngx_helpers_d3d.h",           "ec75224f36ed6580baaf250fa83620405ad81a1d98c2c436654a0c9cf6a6b8ba"),
    @("include/nvsdk_ngx_helpers_dlssd.h",     "include/nvsdk_ngx_helpers_dlssd.h",         "a25fdda925b479dfe869593aeba311ef08793da54e0391f2fe2fe756cbef5532"),
    @("include/nvsdk_ngx_helpers_dlssd_d3d.h", "include/nvsdk_ngx_helpers_dlssd_d3d.h",     "6df9d02c6f47feee3afc8d12340a9f554a7ddce4ddacb1b6a13a42b26c2c6293"),
    @("include/nvsdk_ngx_params.h",            "include/nvsdk_ngx_params.h",                "943bc8cc5cdae03b6303016fbad3183636f2335ae27a2d18776798c3b4efabbc"),
    @("include/nvsdk_ngx_params_dlssd.h",      "include/nvsdk_ngx_params_dlssd.h",          "d41e3ba8f1d72583b55273092c99bf426dc4a0d46de827229e25d5b1325a370e")
)

function Sha256([string]$path) { (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant() }

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$got = 0
foreach ($f in $Files) {
    $local = Join-Path $Dest $f[0]
    if ((Test-Path $local) -and (Sha256 $local) -eq $f[2]) { continue }
    New-Item -ItemType Directory -Force (Split-Path $local) | Out-Null
    $tmp = "$local.download"
    $old = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"  # the progress bar makes Invoke-WebRequest very slow in Windows PowerShell
    try { Invoke-WebRequest -Uri "$Base/$($f[1])" -OutFile $tmp -UseBasicParsing } finally { $ProgressPreference = $old }
    $hash = Sha256 $tmp
    if ($hash -ne $f[2]) {
        Remove-Item $tmp -Force
        throw "$($f[1]) from NVIDIA/DLSS $Tag has SHA256 $hash, expected $($f[2])."
    }
    Move-Item $tmp $local -Force
    $got++
}
"NVIDIA DLSS SDK $Tag in $Dest ($($Files.Count) files checked, $got downloaded)."
