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
- `ghidra/`: headless Ghidra export scripts (string xrefs, references to RVAs, decompilation); see [ghidra/README.md](ghidra/README.md).

Example:

```bash
dumpbin //symbols path/to/mod.dll > syms_raw.txt
python -I tools/re/coff_symbols.py syms_raw.txt > syms.txt
python -I tools/re/disasm.py path/to/mod.dll syms.txt renderHook > renderHook.txt
```
