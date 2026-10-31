// Names engine task classes from their constructors (the engine has no RTTI).
// A task constructor stores its vtable last; slot 6 of a task vtable is GetName(), a stub that returns a string.
// For each constructor: the last vtable it references, the name from slot 6 and the Execute slot (9).
// Args: <out-file> <rva-file>   (one hex constructor RVA per line, optional label after whitespace, '#' starts a comment)
// @category ams2-qvfr

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.symbol.Reference;

public class ExportTaskNames extends GhidraScript {

    private static final int kNameSlot = 6;
    private static final int kExecuteSlot = 9;

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: ExportTaskNames <out-file> <rva-file>");
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
                String label = parts.length > 1 ? parts[1] : "";
                Function ctor = getFunctionContaining(base.add(rva));
                if (ctor == null) {
                    out.printf("ctor rva 0x%x %s: no function%n", rva, label);
                    continue;
                }
                Address vtable = null;
                InstructionIterator it = currentProgram.getListing().getInstructions(ctor.getBody(), true);
                while (it.hasNext()) {
                    Instruction ins = it.next();
                    for (Reference ref : ins.getReferencesFrom()) {
                        Address to = ref.getToAddress();
                        if (isVtable(mem, to)) vtable = to;
                    }
                }
                if (vtable == null) {
                    out.printf("ctor rva 0x%x %s: no vtable%n", rva, label);
                    continue;
                }
                Address nameFn = toAddr(mem.getLong(vtable.add(kNameSlot * 8)));
                Address execFn = toAddr(mem.getLong(vtable.add(kExecuteSlot * 8)));
                out.printf("ctor rva 0x%x vtable rva 0x%x name \"%s\" execute rva 0x%x %s%n", rva,
                        vtable.subtract(base), stubString(mem, nameFn), execFn.subtract(base), label);
            }
        }
        println("written " + args[0]);
    }

    private boolean isExec(Memory mem, Address a) {
        MemoryBlock b = mem.getBlock(a);
        return b != null && b.isExecute();
    }

    // A vtable candidate: a non-executable address whose first ten qwords point to code.
    private boolean isVtable(Memory mem, Address a) {
        if (!a.isMemoryAddress() || isExec(mem, a)) return false;
        try {
            for (int i = 0; i <= kExecuteSlot; i++) {
                if (!isExec(mem, toAddr(mem.getLong(a.add(i * 8))))) return false;
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    // GetName stubs are usually "lea rax, [string]; ret": take the first string referenced by the first few instructions.
    private String stubString(Memory mem, Address fn) {
        try {
            if (getInstructionAt(fn) == null) disassemble(fn);
            Instruction ins = getInstructionAt(fn);
            for (int n = 0; ins != null && n < 4; n++, ins = ins.getNext()) {
                for (Reference ref : ins.getReferencesFrom()) {
                    Address s = ref.getToAddress();
                    if (!s.isMemoryAddress() || isExec(mem, s)) continue;
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < 200; i++) {
                        byte b = mem.getByte(s.add(i));
                        if (b < 0x20 || b > 0x7e) break;
                        sb.append((char) b);
                    }
                    if (sb.length() > 2) return sb.toString();
                }
                if (ins.getFlowType().isTerminal()) break;
            }
        } catch (Exception e) {
            return "? (" + e.getClass().getSimpleName() + ")";
        }
        return "?";
    }
}
