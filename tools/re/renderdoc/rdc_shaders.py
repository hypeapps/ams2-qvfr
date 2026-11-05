"""Dump the shaders, constant buffer layouts and blend state of chosen events of a RenderDoc capture.

Run inside RenderDoc's embedded Python, like rdc_dump.py:

    set RDC_CAP=D:\path\capture.rdc
    set RDC_OUTDIR=D:\path\out
    set RDC_EIDS=27182,27291,37208
    qrenderdoc.exe --python tools\re\renderdoc\rdc_shaders.py

Writes ev_<eid>.txt per event: blend state of RT 0, VS and PS with their constant buffers (variable names; AMS2 values
read back as zeros, see rdc_dump.py), shader resource names and the DXBC disassembly (once per shader).
"""
import os, re, traceback
OUT = os.environ['RDC_OUTDIR']; CAP = os.environ['RDC_CAP']
EIDS = [int(x) for x in os.environ['RDC_EIDS'].split(',')]
log = open(os.path.join(OUT, 'shaders_log.txt'), 'w')
def L(*a): log.write(' '.join(str(x) for x in a) + '\n'); log.flush()
def rid(r):
    m = re.search(r'(\d+)', str(r)); return int(m.group(1)) if m else 0
def fmtvar(v, ind='  '):
    s = []
    if len(v.members):
        for m in v.members: s += fmtvar(m, ind + '  ')
        return ['%s%s:' % (ind, v.name)] + s
    vals = []
    for i in range(v.rows * v.columns):
        try: vals.append('%g' % v.value.f32v[i])
        except Exception: vals.append('?')
    return ['%s%s = %s' % (ind, v.name, ' '.join(vals))]
try:
    import renderdoc as rd
    cap = rd.OpenCaptureFile(); L('open', cap.OpenFile(CAP, '', None))
    st, ctl = cap.OpenCapture(rd.ReplayOptions(), None); L('status', st)
    seen = {}
    targets = ctl.GetDisassemblyTargets(True)
    L('targets', targets)
    for e in EIDS:
        ctl.SetFrameEvent(e, True)
        ps = ctl.GetPipelineState()
        f = open(os.path.join(OUT, 'ev_%d.txt' % e), 'w')
        try:
            for i, b in enumerate(ps.GetColorBlends()[:1]):
                f.write('blend enabled=%s color %s %s %s alpha %s %s %s\n' % (b.enabled, b.colorBlend.source, b.colorBlend.destination, b.colorBlend.operation, b.alphaBlend.source, b.alphaBlend.destination, b.alphaBlend.operation))
        except Exception as ex:
            f.write('blend ? %s\n' % ex)
        for stage, nm in ((rd.ShaderStage.Vertex, 'VS'), (rd.ShaderStage.Pixel, 'PS')):
            refl = ps.GetShaderReflection(stage)
            if refl is None: f.write('%s none\n' % nm); continue
            sid = rid(refl.resourceId)
            f.write('==== %s %d entry %s\n' % (nm, sid, refl.entryPoint))
            for i, cb in enumerate(refl.constantBlocks):
                f.write('-- cbuffer %s (slot %d, %d bytes)\n' % (cb.name, i, cb.byteSize))
                try:
                    bind = ps.GetConstantBlock(stage, i, 0)
                    vars_ = ctl.GetCBufferVariableContents(ps.GetGraphicsPipelineObject(), refl.resourceId, stage, refl.entryPoint, i, bind.descriptor.resource, bind.descriptor.byteOffset, bind.descriptor.byteSize)
                    for v in vars_: f.write('\n'.join(fmtvar(v)) + '\n')
                except Exception as ex:
                    for v in cb.variables: f.write('  %s\n' % v.name)
                    f.write('  (values: %s)\n' % ex)
            for r in refl.readOnlyResources: f.write('-- srv %s\n' % r.name)
            if sid not in seen:
                seen[sid] = e
                f.write(ctl.DisassembleShader(ps.GetGraphicsPipelineObject(), refl, targets[0]))
            else:
                f.write('(disassembly in ev_%d.txt)\n' % seen[sid])
        f.close(); L('ev', e)
    L('done'); ctl.Shutdown()
except Exception:
    L(traceback.format_exc())
log.close(); os._exit(0)
