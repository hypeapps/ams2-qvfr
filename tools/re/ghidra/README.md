# Ghidra scripts

Headless export scripts for the AMS2 executable. They only read the analyzed program; results are text files.

| Script | Args | Output |
|---|---|---|
| `ExportStringXrefs.java` | `<out-file> <needle-file>` | functions that reference strings containing each needle |
| `ExportRefsTo.java` | `<out-file> <rva-file>` | references to each RVA (for example delay-load IAT slots) |
| `DecompileFunctions.java` | `<out-dir> <rva-file>` | one `.c` file per function containing each RVA, with callers and callees |
| `ExportVtables.java` | `<out-file> <rva-file>` | vtable recovered from one known slot: start, slots, constructors that reference it |
| `ExportTaskNames.java` | `<out-file> <rva-file>` | render task class name (slot 6 `GetName`) and `Execute` (slot 9) from a constructor |

Input files take one entry per line; `#` starts a comment. RVA files take a hex RVA and an optional label.
`needles_renderer.txt` is the needle list for the renderer survey (M3).

## Running

Import and auto-analysis (once per exe, takes a long time):

```bat
set GHIDRA_HEADLESS_MAXMEM=12G
<ghidra>\support\analyzeHeadless.bat <project-dir> <project-name> -import <path-to-exe> -max-cpu 8
```

Exports on the analyzed program (`-noanalysis -readOnly` keeps the project unchanged):

```bat
<ghidra>\support\analyzeHeadless.bat <project-dir> <project-name> -process AMS2AVX.exe -noanalysis -readOnly ^
    -scriptPath tools\re\ghidra -postScript ExportStringXrefs.java <out>\xrefs.txt tools\re\ghidra\needles_renderer.txt
```

Several `-postScript` options may follow one another in a single run.
