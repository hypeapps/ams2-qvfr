# Screenshot of the settings GUI (qvfr_config.exe) on one tab, for checking the layout without a person.
# Run it with a test mod folder (--home), never with the game folder: the agent does not run programs from there.
# Example: .\tools\gui_screenshot.ps1 -Exe build\bin\Release\qvfr_config.exe -HomeDir <test>\x64\qvfr -Tab 1 -Out shot.png
# Tabs: 0 Settings, 1 Resolution, 2 Status, 3 Files.
param([string]$Exe, [string]$HomeDir, [int]$Tab, [string]$Out)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class W {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
}
"@
[W]::SetProcessDPIAware() | Out-Null
$p = Start-Process -FilePath $Exe -ArgumentList @("--home", "`"$HomeDir`"", "--tab", "$Tab") -PassThru
Start-Sleep -Milliseconds 2500
$p.Refresh()
$h = $p.MainWindowHandle
$r = New-Object W+RECT
[W]::GetWindowRect($h, [ref]$r) | Out-Null
$bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$dc = $g.GetHdc()
[W]::PrintWindow($h, $dc, 2) | Out-Null
$g.ReleaseHdc($dc)
$bmp.Save($Out)
$p.CloseMainWindow() | Out-Null
Start-Sleep -Milliseconds 500
if (-not $p.HasExited) { $p.Kill() }
"saved $Out"
