"""Builds byte-pattern anchors from RVAs in one PE and finds them in other PEs.

Usage: python anchors.py <reference-pe> <rva-file> <pe> [<pe> ...]
RVA file lines: "<hex-rva> <name>" ('#' starts a comment). The pattern covers whole instructions from the RVA
until it is at least MIN_LEN bytes long; RIP-relative displacements, rel32 branch targets and 64-bit immediates
are wildcarded. If a pattern is not unique in the reference PE it grows (up to MAX_LEN) until it is.
Output per anchor: the pattern and, for each PE, the matching RVAs.
"""
import re
import sys

import capstone
import pefile

MIN_LEN = 24
MAX_LEN = 96


def load(path):
    pe = pefile.PE(path, fast_load=True)
    text = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".text")
    return pe, text.get_data(), text.VirtualAddress


def build_pattern(code, base_rva, rva, min_len):
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    off = rva - base_rva
    raw, mask = bytearray(), bytearray()
    for ins in md.disasm(code[off:off + MAX_LEN + 16], rva):
        b = bytearray(ins.bytes)
        m = bytearray(b"\xff" * len(b))
        wild = []
        if ins.disp_size and ins.disp_offset and any(
                op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP for op in ins.operands):
            wild.append((ins.disp_offset, ins.disp_size))
        if ins.imm_size and ins.imm_offset and (ins.group(capstone.CS_GRP_JUMP) or ins.group(capstone.CS_GRP_CALL)):
            if ins.imm_size == 4:
                wild.append((ins.imm_offset, 4))
        if ins.imm_size == 8:
            wild.append((ins.imm_offset, 8))
        for o, n in wild:
            m[o:o + n] = b"\x00" * n
        raw += b
        mask += m
        if len(raw) >= min_len:
            break
    return bytes(raw), bytes(mask)


def to_regex(raw, mask):
    return re.compile(b"".join(re.escape(bytes([c])) if k else b"." for c, k in zip(raw, mask)), re.DOTALL)


def to_text(raw, mask):
    return " ".join("%02X" % c if k else "??" for c, k in zip(raw, mask))


def find(rx, code, base_rva):
    return [base_rva + m.start() for m in rx.finditer(code)]


def main():
    ref_pe, ref_code, ref_base = load(sys.argv[1])
    targets = [(p, *load(p)[1:]) for p in sys.argv[3:]]
    for ln in open(sys.argv[2], encoding="utf-8"):
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        parts = ln.split(None, 1)
        rva, name = int(parts[0], 16), parts[1] if len(parts) > 1 else ""
        length = MIN_LEN
        while True:
            raw, mask = build_pattern(ref_code, ref_base, rva, length)
            rx = to_regex(raw, mask)
            hits = find(rx, ref_code, ref_base)
            if len(hits) <= 1 or length >= MAX_LEN:
                break
            length += 16
        print("== %s ref rva 0x%x len %d" % (name, rva, len(raw)))
        print("   " + to_text(raw, mask))
        for path, code, base in targets:
            hits = find(rx, code, base)
            shown = ", ".join("0x%x" % h for h in hits[:4]) + (" ..." if len(hits) > 4 else "")
            print("   %-12s %d hit(s): %s" % (path.replace("\\", "/").rsplit("/", 1)[-1], len(hits), shown))


if __name__ == "__main__":
    main()
