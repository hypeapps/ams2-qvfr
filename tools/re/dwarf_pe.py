"""Parse DWARF debug info embedded in a MinGW PE file and dump a readable summary.

Usage: python dwarf_pe.py <pe-file> <out-dir>
Writes: cus.txt (compile units), funcs.txt (functions with params/addresses),
types.txt (struct/class/enum layouts), globals.txt (global variables).
"""
import io
import os
import struct
import sys

import pefile
from elftools.dwarf.dwarfinfo import DWARFInfo, DebugSectionDescriptor, DwarfConfig


def section_map(path):
    pe = pefile.PE(path, fast_load=True)
    data = open(path, "rb").read()
    fh = pe.FILE_HEADER
    strtab_off = fh.PointerToSymbolTable + fh.NumberOfSymbols * 18
    out = {}
    for s in pe.sections:
        raw = s.Name.rstrip(b"\0").decode("latin1")
        if raw.startswith("/"):
            off = strtab_off + int(raw[1:])
            end = data.index(b"\0", off)
            raw = data[off:end].decode("latin1")
        out[raw] = (s, data[s.PointerToRawData: s.PointerToRawData + s.SizeOfRawData])
    return pe, out


def make_dwarf(secs):
    def desc(name):
        if name not in secs:
            return None
        s, blob = secs[name]
        size = min(s.Misc_VirtualSize or len(blob), len(blob))
        return DebugSectionDescriptor(io.BytesIO(blob[:size]), name, None, size, 0)

    return DWARFInfo(
        config=DwarfConfig(little_endian=True, default_address_size=8, machine_arch="x64"),
        debug_info_sec=desc(".debug_info"),
        debug_aranges_sec=desc(".debug_aranges"),
        debug_abbrev_sec=desc(".debug_abbrev"),
        debug_frame_sec=None,
        eh_frame_sec=None,
        debug_str_sec=desc(".debug_str"),
        debug_loc_sec=desc(".debug_loc"),
        debug_ranges_sec=desc(".debug_ranges"),
        debug_line_sec=desc(".debug_line"),
        debug_pubtypes_sec=None,
        debug_pubnames_sec=None,
        debug_addr_sec=desc(".debug_addr"),
        debug_str_offsets_sec=desc(".debug_str_offsets"),
        debug_line_str_sec=desc(".debug_line_str"),
        debug_loclists_sec=desc(".debug_loclists"),
        debug_rnglists_sec=desc(".debug_rnglists"),
        debug_sup_sec=None,
        gnu_debugaltlink_sec=None,
        debug_types_sec=desc(".debug_types"),
    )


def attr(die, name, default=None):
    a = die.attributes.get(name)
    if a is None:
        return default
    v = a.value
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def type_name(die, depth=0):
    if die is None or depth > 12:
        return "void"
    tag = die.tag
    nm = attr(die, "DW_AT_name")
    sub = die.get_DIE_from_attribute("DW_AT_type") if "DW_AT_type" in die.attributes else None
    if tag == "DW_TAG_pointer_type":
        return type_name(sub, depth + 1) + "*"
    if tag in ("DW_TAG_reference_type", "DW_TAG_rvalue_reference_type"):
        return type_name(sub, depth + 1) + "&"
    if tag == "DW_TAG_const_type":
        return "const " + type_name(sub, depth + 1)
    if tag == "DW_TAG_volatile_type":
        return "volatile " + type_name(sub, depth + 1)
    if tag == "DW_TAG_array_type":
        dims = []
        for c in die.iter_children():
            if c.tag == "DW_TAG_subrange_type":
                cnt = attr(c, "DW_AT_count")
                ub = attr(c, "DW_AT_upper_bound")
                dims.append(str(cnt if cnt is not None else (ub + 1 if isinstance(ub, int) else "?")))
        return type_name(sub, depth + 1) + "".join(f"[{d}]" for d in dims)
    if tag == "DW_TAG_subroutine_type":
        return "fnptr"
    if nm:
        return nm
    if tag == "DW_TAG_typedef":
        return type_name(sub, depth + 1)
    return tag.replace("DW_TAG_", "")


def qualname(die):
    parts = []
    d = die
    while d is not None:
        nm = attr(d, "DW_AT_name")
        if d.tag in ("DW_TAG_namespace", "DW_TAG_structure_type", "DW_TAG_class_type", "DW_TAG_subprogram",
                     "DW_TAG_variable", "DW_TAG_enumeration_type", "DW_TAG_union_type", "DW_TAG_typedef"):
            parts.append(nm or "(anon)")
        d = d.get_parent()
    return "::".join(reversed(parts))


def main():
    path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    pe, secs = section_map(path)
    base = pe.OPTIONAL_HEADER.ImageBase
    dw = make_dwarf(secs)

    cus, funcs, types, gvars = [], [], [], []
    for cu in dw.iter_CUs():
        top = cu.get_top_DIE()
        cuname = attr(top, "DW_AT_name", "?")
        cus.append(f"{cuname}  (comp_dir={attr(top, 'DW_AT_comp_dir', '')})")
        skip = any(k in cuname.replace("\\", "/") for k in ("/vendor/", "minhook", "libcxx", "llvm", "mingw-w64-crt", "compiler-rt"))
        for die in cu.iter_DIEs():
            try:
                if die.tag == "DW_TAG_subprogram" and not skip:
                    spec = die
                    if "DW_AT_specification" in die.attributes:
                        spec = die.get_DIE_from_attribute("DW_AT_specification")
                    elif "DW_AT_abstract_origin" in die.attributes:
                        spec = die.get_DIE_from_attribute("DW_AT_abstract_origin")
                    low = attr(die, "DW_AT_low_pc")
                    if low is None:
                        continue
                    params = []
                    for c in die.iter_children():
                        if c.tag == "DW_TAG_formal_parameter":
                            pc = c
                            if "DW_AT_abstract_origin" in c.attributes:
                                pc = c.get_DIE_from_attribute("DW_AT_abstract_origin")
                            t = pc.get_DIE_from_attribute("DW_AT_type") if "DW_AT_type" in pc.attributes else None
                            params.append(f"{type_name(t)} {attr(pc, 'DW_AT_name', '')}".strip())
                    rt = spec.get_DIE_from_attribute("DW_AT_type") if "DW_AT_type" in spec.attributes else None
                    line = attr(spec, "DW_AT_decl_line", "")
                    funcs.append((low - base if low >= base else low, f"{type_name(rt)} {qualname(spec)}({', '.join(params)})  [{os.path.basename(cuname)}:{line}]"))
                elif die.tag in ("DW_TAG_structure_type", "DW_TAG_class_type", "DW_TAG_enumeration_type") and not skip:
                    if attr(die, "DW_AT_declaration"):
                        continue
                    qn = qualname(die)
                    if not (qn.startswith("lmuqv") or qn.startswith("(anon)") or "::" not in qn):
                        continue
                    if qn.startswith("std") or qn.startswith("__"):
                        continue
                    lines = [f"{die.tag.replace('DW_TAG_', '')} {qn} size={attr(die, 'DW_AT_byte_size')}  [{os.path.basename(cuname)}]"]
                    for c in die.iter_children():
                        if c.tag == "DW_TAG_member":
                            t = c.get_DIE_from_attribute("DW_AT_type") if "DW_AT_type" in c.attributes else None
                            lines.append(f"    +{attr(c, 'DW_AT_data_member_location', '?')}: {type_name(t)} {attr(c, 'DW_AT_name', '')}")
                        elif c.tag == "DW_TAG_enumerator":
                            lines.append(f"    {attr(c, 'DW_AT_name')} = {attr(c, 'DW_AT_const_value')}")
                    types.append("\n".join(lines))
                elif die.tag == "DW_TAG_variable" and not skip and die.get_parent().tag in ("DW_TAG_compile_unit", "DW_TAG_namespace"):
                    t = die.get_DIE_from_attribute("DW_AT_type") if "DW_AT_type" in die.attributes else None
                    cv = attr(die, "DW_AT_const_value")
                    gvars.append(f"{type_name(t)} {qualname(die)}{(' = ' + repr(cv)[:200]) if cv is not None else ''}  [{os.path.basename(cuname)}:{attr(die, 'DW_AT_decl_line', '')}]")
            except Exception as e:  # keep going on odd DIEs
                continue

    open(os.path.join(outdir, "cus.txt"), "w", encoding="utf-8").write("\n".join(cus))
    funcs.sort()
    open(os.path.join(outdir, "funcs.txt"), "w", encoding="utf-8").write("\n".join(f"{a:08x} {s}" for a, s in funcs))
    seen = []
    for t in types:
        if t not in seen:
            seen.append(t)
    open(os.path.join(outdir, "types.txt"), "w", encoding="utf-8").write("\n\n".join(seen))
    open(os.path.join(outdir, "globals.txt"), "w", encoding="utf-8").write("\n".join(sorted(set(gvars))))
    print(len(cus), "CUs,", len(funcs), "funcs,", len(seen), "types,", len(set(gvars)), "globals")


if __name__ == "__main__":
    main()
