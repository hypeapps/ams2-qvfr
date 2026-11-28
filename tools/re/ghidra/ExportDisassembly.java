// Lists the instructions at the given RVAs, disassembling bytes that auto-analysis left undefined (in memory only;
// run with -readOnly so the project stays unchanged).
// Args: <out-file> <rva-file>   (one entry per line: hex RVA, optional instruction count (default 40), optional label;
// '#' starts a comment). An RVA may point inside a function; listing starts at that exact address.
// @category ams2-qvfr

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.mem.MemoryAccessException;

public class ExportDisassembly extends GhidraScript {

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: ExportDisassembly <out-file> <rva-file>");
            return;
        }
        Address base = currentProgram.getImageBase();
        try (PrintWriter out = new PrintWriter(new File(args[0]), StandardCharsets.UTF_8)) {
            for (String line : Files.readAllLines(new File(args[1]).toPath(), StandardCharsets.UTF_8)) {
                line = line.strip();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\\s+", 3);
                long rva = Long.parseUnsignedLong(parts[0].replaceFirst("^0[xX]", ""), 16);
                int count = 40;
                String label = "";
                if (parts.length > 1) {
                    try {
                        count = Integer.parseInt(parts[1]);
                        label = parts.length > 2 ? parts[2] : "";
                    } catch (NumberFormatException e) {
                        label = line.substring(parts[0].length()).strip();
                    }
                }
                Address a = base.add(rva);
                Function f = getFunctionContaining(a);
                out.printf("== rva 0x%x %s (function %s)%n", rva, label,
                        f == null ? "none" : String.format("%s at rva 0x%x", f.getName(), f.getEntryPoint().subtract(base)));
                for (int i = 0; i < count && a != null; i++) {
                    Instruction ins = getInstructionAt(a);
                    if (ins == null) {
                        disassemble(a);
                        ins = getInstructionAt(a);
                    }
                    if (ins == null) {
                        out.printf("  0x%x  <no instruction>%n", a.subtract(base));
                        break;
                    }
                    out.printf("  0x%x  %-24s %s%n", a.subtract(base), hex(ins), ins);
                    a = ins.getMaxAddress().next();
                }
            }
        }
        println("written " + args[0]);
    }

    private static String hex(Instruction ins) {
        StringBuilder sb = new StringBuilder();
        try {
            for (byte b : ins.getBytes()) sb.append(String.format("%02X", b & 0xff));
        } catch (MemoryAccessException e) {
            sb.append("??");
        }
        return sb.toString();
    }
}
