// Lists the references to the given RVAs (for example delay-load IAT slots) and the functions they come from.
// Args: <out-file> <rva-file>   (one hex RVA per line, optional label after whitespace, '#' starts a comment)
// @category ams2-qvfr

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.Reference;

public class ExportRefsTo extends GhidraScript {

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: ExportRefsTo <out-file> <rva-file>");
            return;
        }
        Address base = currentProgram.getImageBase();
        try (PrintWriter out = new PrintWriter(args[0], StandardCharsets.UTF_8)) {
            for (String line : Files.readAllLines(new File(args[1]).toPath(), StandardCharsets.UTF_8)) {
                line = line.strip();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\\s+", 2);
                long rva = Long.parseUnsignedLong(parts[0].replaceFirst("^0[xX]", ""), 16);
                out.printf("== rva 0x%x %s%n", rva, parts.length > 1 ? parts[1] : "");
                for (Reference ref : getReferencesTo(base.add(rva))) {
                    Address from = ref.getFromAddress();
                    Function f = getFunctionContaining(from);
                    out.printf("  %s from rva 0x%x%s%n", ref.getReferenceType(), from.subtract(base),
                            f == null ? " (no function)"
                                      : String.format(" in %s (entry rva 0x%x)", f.getName(),
                                                      f.getEntryPoint().subtract(base)));
                }
            }
        }
        println("written " + args[0]);
    }
}
