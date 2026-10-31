"""Annotated x64 disassembly of named functions from a PE with a COFF symbol table.

Usage: python disasm.py <pe> <demangled-symbols.txt> <name-substring> [...]
Symbols file lines: "SECTn <hex-offset> F|D <name>" (offsets relative to section start).
"""
import re
import sys

import capstone
import pefile


def load(pe_path, sym_path):
    pe = pefile.PE(pe_path)
    secs = pe.sections
    syms = []
    for ln in open(sym_path, encoding="utf-8"):
        m = re.match(r"SECT(\d+) ([0-9a-f]+) ([FD]) (.*)", ln.rstrip("\n"))
        if not m:
            continue
        idx, off, kind, name = int(m[1]), int(m[2], 16), m[3], m[4]
        rva = secs[idx - 1].VirtualAddress + off
        syms.append((rva, kind, name))
    syms.sort()
    imports = {}
    for d in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        for i in d.imports:
            imports[i.address - pe.OPTIONAL_HEADER.ImageBase] = f"{d.dll.decode()}!{(i.name or b'?').decode()}"
    return pe, syms, imports


def sym_at(syms, rva):
    lo, hi = 0, len(syms)
    while lo < hi:
        mid = (lo + hi) // 2
        if syms[mid][0] <= rva:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return None
    s = syms[lo - 1]
    return s if rva - s[0] < 0x4000 else None


def cstr(pe, rva):
    try:
        b = pe.get_data(rva, 200)
    except Exception:
        return None
    end = b.find(b"\0")
    if end < 4:
        return None
    s = b[:end]
    if all(32 <= c < 127 or c in (9, 10) for c in s):
        return s.decode()
    return None


def main():
    pe, syms, imports = load(sys.argv[1], sys.argv[2])
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = False
    names = sys.argv[3:]
    for i, (rva, kind, name) in enumerate(syms):
        if kind != "F" or not any(n in name for n in names):
            continue
        end = next((r for r, k, _ in syms[i + 1:] if r > rva), rva + 0x2000)
        code = pe.get_data(rva, end - rva)
        print(f"\n===== {name}  rva={rva:#x} size={end - rva:#x}")
        for ins in md.disasm(code, rva):
            note = ""
            m = re.search(r"\[rip ([+-]) (0x[0-9a-f]+)\]", ins.op_str)
            target = None
            if m:
                target = ins.address + ins.size + (int(m[2], 16) * (1 if m[1] == "+" else -1))
            elif ins.mnemonic in ("call", "jmp") and ins.op_str.startswith("0x"):
                target = int(ins.op_str, 16)
            if target is not None:
                if target in imports:
                    note = imports[target]
                else:
                    s = sym_at(syms, target)
                    st = cstr(pe, target)
                    if st:
                        note = repr(st[:120])
                    elif s:
                        note = s[2] + (f"+{target - s[0]:#x}" if target != s[0] else "")
            if ins.mnemonic.startswith("j") and ins.mnemonic != "jmp" and not note:
                note = ""
            print(f"  {ins.address:08x}  {ins.mnemonic:7} {ins.op_str}" + (f"    ; {note}" if note else ""))


if __name__ == "__main__":
    main()
