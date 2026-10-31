// Lists instructions whose operands contain one of the given scalars (for example struct field offsets
// such as [reg + 0x27c30]), grouped by function.
// Args: <out-file> <scalar-file>   (one hex value per line, optional label after whitespace, '#' starts a comment)
// @category ams2-qvfr

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.scalar.Scalar;

public class ExportScalarUses extends GhidraScript {

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: ExportScalarUses <out-file> <scalar-file>");
            return;
        }
        Map<Long, String> wanted = new HashMap<>();
        for (String line : Files.readAllLines(new File(args[1]).toPath(), StandardCharsets.UTF_8)) {
            line = line.strip();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split("\\s+", 2);
            wanted.put(Long.parseUnsignedLong(parts[0].replaceFirst("^0[xX]", ""), 16), parts.length > 1 ? parts[1] : "");
        }
        Address base = currentProgram.getImageBase();
        // value -> function entry rva -> instruction lines
        Map<Long, TreeMap<Long, List<String>>> hits = new TreeMap<>();
        InstructionIterator it = currentProgram.getListing().getInstructions(true);
        while (it.hasNext() && !monitor.isCancelled()) {
            Instruction ins = it.next();
            for (int op = 0; op < ins.getNumOperands(); op++) {
                for (Object o : ins.getOpObjects(op)) {
                    if (!(o instanceof Scalar)) continue;
                    long v = ((Scalar) o).getUnsignedValue();
                    if (!wanted.containsKey(v)) continue;
                    Function f = getFunctionContaining(ins.getAddress());
                    long entry = f == null ? -1 : f.getEntryPoint().subtract(base);
                    hits.computeIfAbsent(v, k -> new TreeMap<>())
                        .computeIfAbsent(entry, k -> new ArrayList<>())
                        .add(String.format("0x%x  %s", ins.getAddress().subtract(base), ins));
                }
            }
        }
        try (PrintWriter out = new PrintWriter(args[0], StandardCharsets.UTF_8)) {
            for (Map.Entry<Long, String> w : new TreeMap<>(wanted).entrySet()) {
                TreeMap<Long, List<String>> byFn = hits.getOrDefault(w.getKey(), new TreeMap<>());
                out.printf("== 0x%x %s (%d functions)%n", w.getKey(), w.getValue(), byFn.size());
                for (Map.Entry<Long, List<String>> e : byFn.entrySet()) {
                    Function f = e.getKey() < 0 ? null : getFunctionAt(base.add(e.getKey()));
                    out.printf("  fn rva 0x%x %s%n", e.getKey(), f == null ? "(none)" : f.getName());
                    for (String s : e.getValue()) out.println("      " + s);
                }
            }
        }
        println("written " + args[0]);
    }
}
