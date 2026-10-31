"""Turn `dumpbin /symbols` output into a demangled symbol list for disasm.py.

Usage: python coff_symbols.py <dumpbin-symbols.txt> > symbols.txt
Output lines: "SECTn <hex offset in section> F|D <demangled name>". F = function, D = data.
Skips libc++/libunwind/MinHook/CRT noise. Needs `itanium_demangler` (pip).
"""
import re
import sys

from itanium_demangler import parse

NOISE = re.compile(r"(NSt3__1|itanium_demangle|libunwind|__unw|^__|^\.|^_?_?mingw|^_Unwind|^MH_|^\?)")
LINE = re.compile(r"^[0-9A-F]+\s+([0-9A-F]{8})\s+(SECT\w+|UNDEF|ABS)\s+\S+\s*(\(\))?\s+(\w+)\s+\|\s+(\S+)")


def main():
    rows = []
    for ln in open(sys.argv[1], encoding="latin1"):
        m = LINE.match(ln)
        if not m:
            continue
        addr, sect, fn, _scope, name = m.groups()
        if NOISE.search(name) or not sect.startswith("SECT"):
            continue
        try:
            shown = str(parse(name)) if name.startswith("_Z") else name
        except Exception:
            shown = name
        rows.append((sect, int(addr, 16), "F" if fn else "D", shown))
    rows.sort()
    for sect, addr, kind, name in rows:
        print(f"{sect} {addr:08x} {kind} {name}")


if __name__ == "__main__":
    main()
