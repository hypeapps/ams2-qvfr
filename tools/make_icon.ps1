# Generates the settings GUI icon (src/config/qvfr_config.ico): a minimal mark of the mod's idea. A dark rounded tile holds
# a thin light frame (the outer view, the whole field of view) and a solid red square off the middle, up and right (the
# focus view, following the gaze).
# Each size is drawn natively and snapped to whole pixels, so 16 px stays crisp. The 256 entry is a PNG (as Windows
# expects), the smaller ones classic 32-bit DIBs, which every icon reader understands.
# Example: .	ools\make_icon.ps1 -Out src\config\qvfr_config.ico
param([string]$Out = (Join-Path $PSScriptRoot "..\src\config\qvfr_config.ico"))
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

function C([int]$r, [int]$g, [int]$b, [int]$a = 255) { [System.Drawing.Color]::FromArgb($a, $r, $g, $b) }

# Rounded rectangle path (all in pixels).
function RoundRect([float]$x, [float]$y, [float]$w, [float]$h, [float]$r) {
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    if ($r -le 0) { $p.AddRectangle((New-Object System.Drawing.RectangleF $x, $y, $w, $h)); return , $p }
    $d = 2 * $r
    $p.AddArc($x, $y, $d, $d, 180, 90); $p.AddArc($x + $w - $d, $y, $d, $d, 270, 90)
    $p.AddArc($x + $w - $d, $y + $h - $d, $d, $d, 0, 90); $p.AddArc($x, $y + $h - $d, $d, $d, 90, 90)
    $p.CloseFigure()
    return , $p
}

function Png([int]$s) {
    $bmp = New-Object System.Drawing.Bitmap $s, $s, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.Clear([System.Drawing.Color]::Transparent)
    # Tile: the whole icon, rounded; a faint light edge from 32 px.
    $g.FillPath((New-Object System.Drawing.SolidBrush (C 22 22 25)), (RoundRect 0 0 $s $s ([Math]::Round($s * 0.22))))
    if ($s -ge 32) {
        $e = [Math]::Max(1, [Math]::Round($s / 64))
        $g.DrawPath((New-Object System.Drawing.Pen (C 255 255 255 28), $e),
                    (RoundRect ($e / 2) ($e / 2) ($s - $e) ($s - $e) ([Math]::Round($s * 0.22) - $e / 2)))
    }

    # Outer view: a thin light frame, its edges on whole pixels.
    $m = [Math]::Round($s * 3 / 16)               # margin to the frame's outer edge
    $t = [Math]::Max(1, [Math]::Round($s / 18))   # frame thickness
    $g.DrawPath((New-Object System.Drawing.Pen (C 236 236 240), $t),
                (RoundRect ($m + $t / 2) ($m + $t / 2) ($s - 2 * $m - $t) ($s - 2 * $m - $t) ([Math]::Round($s / 16))))

    # Focus view: a red square in the frame's top right corner, the same gap to the frame above and on the right.
    $f = [Math]::Round($s / 4)
    $gap = [Math]::Max(1, [Math]::Round($s / 24))
    $g.FillRectangle((New-Object System.Drawing.SolidBrush (C 226 35 26)), ($s - $m - $t - $gap - $f), ($m + $t + $gap), $f, $f)
    $g.Dispose()
    $ms = New-Object System.IO.MemoryStream
    if ($s -ge 256) {
        $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    } else {
        # Classic 32-bit DIB entry: BITMAPINFOHEADER with the doubled height, BGRA rows bottom-up, then an empty AND mask.
        $w = New-Object System.IO.BinaryWriter $ms
        $w.Write([uint32]40); $w.Write([int32]$s); $w.Write([int32](2 * $s)); $w.Write([uint16]1); $w.Write([uint16]32)
        $w.Write([uint32]0); $w.Write([uint32]0); $w.Write([int32]0); $w.Write([int32]0); $w.Write([uint32]0); $w.Write([uint32]0)
        $data = $bmp.LockBits((New-Object System.Drawing.Rectangle 0, 0, $s, $s), [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                              [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
        $row = New-Object byte[] (4 * $s)
        for ($y = $s - 1; $y -ge 0; $y--) {
            [System.Runtime.InteropServices.Marshal]::Copy([IntPtr]($data.Scan0.ToInt64() + $y * $data.Stride), $row, 0, $row.Length)
            $w.Write($row)
        }
        $bmp.UnlockBits($data)
        $w.Write((New-Object byte[] ([Math]::Ceiling($s / 32.0) * 4 * $s)))
        $w.Flush()
    }
    $bmp.Dispose()
    return , $ms.ToArray()
}

$sizes = @(16, 20, 24, 32, 40, 48, 64, 256)
$images = @($sizes | ForEach-Object { , (Png $_) })
$ms = New-Object System.IO.MemoryStream
$w = New-Object System.IO.BinaryWriter $ms
$w.Write([uint16]0); $w.Write([uint16]1); $w.Write([uint16]$sizes.Count)  # ICONDIR
$offset = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {  # ICONDIRENTRY; 0 means 256
    $d = if ($sizes[$i] -ge 256) { 0 } else { $sizes[$i] }
    $w.Write([byte]$d); $w.Write([byte]$d); $w.Write([byte]0); $w.Write([byte]0)
    $w.Write([uint16]1); $w.Write([uint16]32)
    $w.Write([uint32]$images[$i].Length); $w.Write([uint32]$offset)
    $offset += $images[$i].Length
}
foreach ($img in $images) { $w.Write($img) }
$w.Flush()
[System.IO.File]::WriteAllBytes([System.IO.Path]::GetFullPath($Out), $ms.ToArray())
"saved $Out ($($ms.Length) B, sizes $($sizes -join ', '))"
