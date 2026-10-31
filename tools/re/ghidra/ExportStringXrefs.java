// Lists the functions that reference strings containing the given needles.
// Searches raw memory for each needle, so it does not depend on the string analyzer.
// Args: <out-file> <needle-file>   (one needle per line, '#' starts a comment)
// @category ams2-qvfr

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.symbol.Reference;

public class ExportStringXrefs extends GhidraScript {

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: ExportStringXrefs <out-file> <needle-file>");
            return;
        }
        List<String> needles = new ArrayList<>();
        for (String line : Files.readAllLines(new File(args[1]).toPath(), StandardCharsets.UTF_8)) {
            line = line.strip();
            if (!line.isEmpty() && !line.startsWith("#")) needles.add(line);
        }
        Memory mem = currentProgram.getMemory();
        Address base = currentProgram.getImageBase();
        try (PrintWriter out = new PrintWriter(args[0], StandardCharsets.UTF_8)) {
            for (String needle : needles) {
                out.println("== " + needle);
                byte[] pattern = needle.getBytes(StandardCharsets.US_ASCII);
                TreeSet<Address> starts = new TreeSet<>();
                for (MemoryBlock block : mem.getBlocks()) {
                    if (!block.isInitialized() || block.isExecute()) continue;
                    Address at = block.getStart();
                    while (at != null && at.compareTo(block.getEnd()) <= 0 && !monitor.isCancelled()) {
                        Address hit = mem.findBytes(at, block.getEnd(), pattern, null, true, monitor);
                        if (hit == null) break;
                        starts.add(stringStart(mem, hit, block.getStart()));
                        at = hit.add(1);
                    }
                }
                for (Address s : starts) {
                    out.printf("  string %s rva 0x%x \"%s\"%n", s, s.subtract(base), readString(mem, s));
                    for (Reference ref : getReferencesTo(s)) {
                        Address from = ref.getFromAddress();
                        Function f = getFunctionContaining(from);
                        if (f != null) {
                            out.printf("    ref rva 0x%x in %s (entry rva 0x%x)%n", from.subtract(base), f.getName(),
                                    f.getEntryPoint().subtract(base));
                        } else {
                            out.printf("    ref rva 0x%x (no function)%n", from.subtract(base));
                        }
                    }
                }
            }
        }
        println("written " + args[0]);
    }

    private Address stringStart(Memory mem, Address hit, Address blockStart) throws Exception {
        Address a = hit;
        for (int i = 0; i < 512 && a.compareTo(blockStart) > 0; i++) {
            byte prev = mem.getByte(a.subtract(1));
            if (prev < 0x20 || prev > 0x7e) break;
            a = a.subtract(1);
        }
        return a;
    }

    private String readString(Memory mem, Address a) throws Exception {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 200; i++) {
            byte b = mem.getByte(a.add(i));
            if (b < 0x20 || b > 0x7e) break;
            sb.append((char) b);
        }
        return sb.toString();
    }
}
