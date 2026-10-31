// Decompiles the functions that contain the given RVAs, with their callers and callees.
// An RVA may point inside a function (for example a return address from a stack trace).
// Args: <out-dir> <rva-file>   (one hex RVA per line, optional label after whitespace, '#' starts a comment)
// @category ams2-qvfr

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.HashSet;
import java.util.Set;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;

public class DecompileFunctions extends GhidraScript {

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: DecompileFunctions <out-dir> <rva-file>");
            return;
        }
        File outDir = new File(args[0]);
        outDir.mkdirs();
        Address base = currentProgram.getImageBase();

        DecompInterface ifc = new DecompInterface();
        ifc.setOptions(new DecompileOptions());
        ifc.openProgram(currentProgram);

        Set<Address> done = new HashSet<>();
        for (String line : Files.readAllLines(new File(args[1]).toPath(), StandardCharsets.UTF_8)) {
            line = line.strip();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] parts = line.split("\\s+", 2);
            long rva = Long.parseUnsignedLong(parts[0].replaceFirst("^0[xX]", ""), 16);
            String label = parts.length > 1 ? parts[1] : "";
            Function f = getFunctionContaining(base.add(rva));
            if (f == null) {
                println(String.format("rva 0x%x: no function", rva));
                continue;
            }
            if (!done.add(f.getEntryPoint())) continue;
            long entryRva = f.getEntryPoint().subtract(base);
            File file = new File(outDir, String.format("%08x.c", entryRva));
            try (PrintWriter out = new PrintWriter(file, StandardCharsets.UTF_8)) {
                out.printf("// %s entry rva 0x%x size %d, asked rva 0x%x %s%n", f.getName(), entryRva,
                        f.getBody().getNumAddresses(), rva, label);
                for (Function c : f.getCallingFunctions(monitor)) {
                    out.printf("// caller %s %s%n", where(c, base), c.getName());
                }
                for (Function c : f.getCalledFunctions(monitor)) {
                    out.printf("// callee %s %s%n", where(c, base), c.getName());
                }
                DecompileResults r = ifc.decompileFunction(f, 180, monitor);
                if (r.decompileCompleted()) {
                    out.println(r.getDecompiledFunction().getC());
                } else {
                    out.println("// decompile failed: " + r.getErrorMessage());
                }
            }
        }
        ifc.dispose();
        println("written " + outDir);
    }

    // Imported functions live in the EXTERNAL space and have no RVA.
    private static String where(Function f, Address base) {
        Address a = f.getEntryPoint();
        return a.isMemoryAddress() ? String.format("rva 0x%x", a.subtract(base)) : "external";
    }
}
