// Recovers vtables (the engine has no RTTI) from one known slot each.
// From the slot it walks back to the nearest referenced address (the vtable start that constructors store),
// then lists the slots up to the next referenced address or the first non-code pointer, and the code that
// references the start (constructors and destructors).
// Args: <out-file> <rva-file>   (one hex slot RVA per line, optional label after whitespace, '#' starts a comment)
// @category ams2-qvfr

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.symbol.Reference;

public class ExportVtables extends GhidraScript {

    private static final int kMaxSlots = 256;

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: ExportVtables <out-file> <rva-file>");
            return;
        }
        Address base = currentProgram.getImageBase();
        Memory mem = currentProgram.getMemory();
        try (PrintWriter out = new PrintWriter(args[0], StandardCharsets.UTF_8)) {
            for (String line : Files.readAllLines(new File(args[1]).toPath(), StandardCharsets.UTF_8)) {
                line = line.strip();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\\s+", 2);
                long rva = Long.parseUnsignedLong(parts[0].replaceFirst("^0[xX]", ""), 16);
                Address slot = base.add(rva);

                Address start = slot;
                for (int i = 0; i < kMaxSlots && !referenced(start); i++) {
                    Address prev = start.subtract(8);
                    if (!isCodePointer(mem, prev)) break;
                    start = prev;
                }
                out.printf("== vtable rva 0x%x (slot rva 0x%x is index %d) %s%n", start.subtract(base), rva,
                        slot.subtract(start) / 8, parts.length > 1 ? parts[1] : "");
                for (Reference ref : getReferencesTo(start)) {
                    Function f = getFunctionContaining(ref.getFromAddress());
                    out.printf("  ref from rva 0x%x%s%n", ref.getFromAddress().subtract(base),
                            f == null ? "" : String.format(" in %s (entry rva 0x%x)", f.getName(),
                                                           f.getEntryPoint().subtract(base)));
                }
                Address a = start;
                for (int i = 0; i < kMaxSlots && isCodePointer(mem, a) && (i == 0 || !referenced(a)); i++) {
                    Address target = toAddr(mem.getLong(a));
                    Function f = getFunctionAt(target);
                    out.printf("  [%3d] +0x%03x -> rva 0x%x %s%n", i, i * 8, target.subtract(base),
                            f == null ? "" : f.getName());
                    a = a.add(8);
                }
            }
        }
        println("written " + args[0]);
    }

    private boolean referenced(Address a) {
        return getReferencesTo(a).length > 0;
    }

    private boolean isCodePointer(Memory mem, Address a) {
        try {
            Address target = toAddr(mem.getLong(a));
            MemoryBlock b = mem.getBlock(target);
            return b != null && b.isExecute();
        } catch (Exception e) {
            return false;
        }
    }
}
