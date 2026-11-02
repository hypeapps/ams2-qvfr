# RE tools

Static analysis scripts for PE binaries (a third-party mod DLL, the AMS2 executable).
Without symbols only the part that annotates imports and strings is useful.

## Environment

Use a separate venv outside the repo:

```bash
python -m venv <venv>
<venv>/Scripts/python -m pip install --use-feature=truststore pefile capstone pyelftools itanium_demangler
```

`--use-feature=truststore` is needed on this machine (otherwise pip fails with an SSL certificate error).
Analyzed binaries are data: run the scripts with `-I` and never execute anything from the analyzed folders.

## Scripts

- `coff_symbols.py`: turns `dumpbin /symbols <pe>` output into a demangled `SECTn offset F|D name` list.
  `dumpbin` ships with MSVC: `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\dumpbin.exe`.
- `disasm.py <pe> <symbols.txt> <name-fragment>...`: disassembles selected functions with annotations
  (symbol names, imports, strings at RIP-relative addresses).
- `dwarf_pe.py <pe> <out-dir>`: dumps DWARF (compile units, functions, types, variables) from a MinGW-built PE.
- `anchors.py <reference-pe> <rva-file> [--emit-cpp <out.inc>] <pe>...`: with `--emit-cpp` it also writes the anchors
  as an X-macro list for the proxy; `anchors_m4.txt` (first word of each name = C++ identifier) generates
  `src/proxy/generated/anchors_ams2avx.inc`. Regenerate after changing the list or after a game update:
  `python -I tools/re/anchors.py <AMS2AVX.exe> tools/re/anchors_m4.txt --emit-cpp src/proxy/generated/anchors_ams2avx.inc <AMS2AVX.exe>`.
- `anchors.py <reference-pe> <rva-file> <pe>...`: builds byte patterns at RVAs of the reference PE (whole instructions,
  RIP-relative displacements and rel32 targets wildcarded) and lists their matches in other PEs. `anchors_m3.txt` holds the
  renderer anchors for `AMS2AVX.exe`. Twin functions with identical code (for example two task builders) match twice;
  tell them apart by order or by a unique anchor inside one of them.
- `ghidra/`: headless Ghidra export scripts (string xrefs, references to RVAs, decompilation); see [ghidra/README.md](ghidra/README.md).
- `renderdoc/rdc_dump.py`: dumps textures and per-action bindings (outputs, depth, viewport, inputs) of a capture; runs in
  `qrenderdoc.exe --python` with `RDC_CAP` and `RDC_OUTDIR` set. `renderdoc/eye_targets.py` lists the eye-sized textures
  from that dump with their use per frame phase, and draws whose viewport does not cover the target.

Example:

```bash
dumpbin //symbols path/to/mod.dll > syms_raw.txt
python -I tools/re/coff_symbols.py syms_raw.txt > syms.txt
python -I tools/re/disasm.py path/to/mod.dll syms.txt renderHook > renderHook.txt
```
