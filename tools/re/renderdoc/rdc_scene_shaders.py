"""Collect every shader used by the draws of chosen passes of a RenderDoc capture, with signatures and bindings.

Run inside RenderDoc's embedded Python, like rdc_dump.py:

    set RDC_CAP=D:\\path\\capture.rdc
    set RDC_OUTDIR=D:\\path\\out
    set RDC_RANGES=22817-51949
    set RDC_PASSES=gbuf:15415;fwd:15454;pre1:74038;pre2:74024;light:15445
    qrenderdoc.exe --python tools\\re\\renderdoc\\rdc_scene_shaders.py

RDC_PASSES maps a label to a render target id; a draw belongs to the pass whose target it writes.
Writes to RDC_OUTDIR:
- draws.tsv: eid, pass, indices, instances and the VS/HS/DS/GS/PS ids of every matching draw;
- shaders/<stage>_<id>.txt: input and output signatures, constant buffers with bind slots and variable offsets,
  shader resources and samplers with bind slots, and the DXBC disassembly;
- shaders/<stage>_<id>.dxbc: the shader bytecode as the game created it;
- scene_log.txt: progress and errors.
"""
import os
import re
import traceback

OUT = os.environ['RDC_OUTDIR']
CAP = os.environ['RDC_CAP']
RANGES = [tuple(int(x) for x in r.split('-')) for r in os.environ['RDC_RANGES'].split(';')]
PASSES = {}
for p in os.environ['RDC_PASSES'].split(';'):
    k, v = p.split(':')
    PASSES[int(v)] = k
os.makedirs(os.path.join(OUT, 'shaders'), exist_ok=True)
log = open(os.path.join(OUT, 'scene_log.txt'), 'w')


def L(*a):
    log.write(' '.join(str(x) for x in a) + '\n')
    log.flush()


def rid(r):
    m = re.search(r'(\d+)', str(r))
    return int(m.group(1)) if m else 0


def walk(xs):
    for x in xs:
        yield x
        yield from walk(x.children)


def attr(o, *names, default='?'):
    for n in names:
        if hasattr(o, n):
            return getattr(o, n)
    return default


def fmt_sig(sig):
    rows = []
    for s in sig:
        rows.append('  %s%d reg %s sv %s comps %s mask 0x%x used 0x%x' % (
            s.semanticName, s.semanticIndex, s.regIndex, str(s.systemValue).replace('ShaderBuiltin.', ''),
            s.compCount, s.regChannelMask, s.channelUsedMask))
    return rows


def fmt_vars(vs, base, ind):
    rows = []
    for v in vs:
        t = v.type
        off = base + v.byteOffset
        members = attr(t, 'members', default=[])
        rows.append('%s%-32s off %5d reg %4d rows %d cols %d elems %d' % (
            ind, v.name, off, off // 16, attr(t, 'rows', default=0), attr(t, 'columns', default=0),
            attr(t, 'elements', default=0)))
        if members:
            rows += fmt_vars(members, off, ind + '  ')
    return rows


try:
    import renderdoc as rd
    cap = rd.OpenCaptureFile()
    L('open', cap.OpenFile(CAP, '', None))
    st, ctl = cap.OpenCapture(rd.ReplayOptions(), None)
    L('status', st)
    target = ctl.GetDisassemblyTargets(True)[0]
    F = rd.ActionFlags
    stages = [('vs', rd.ShaderStage.Vertex), ('hs', rd.ShaderStage.Hull), ('ds', rd.ShaderStage.Domain),
              ('gs', rd.ShaderStage.Geometry), ('ps', rd.ShaderStage.Pixel)]
    acts = [a for a in walk(ctl.GetRootActions()) if a.flags & F.Drawcall
            and any(lo <= a.eventId <= hi for lo, hi in RANGES)]
    seen = set()
    n = 0
    with open(os.path.join(OUT, 'draws.tsv'), 'w') as dt:
        dt.write('eid\tpass\tidx\tinst\tvs\ths\tds\tgs\tps\n')
        for a in acts:
            label = None
            for o in a.outputs:
                if rid(o) in PASSES:
                    label = PASSES[rid(o)]
                    break
            if label is None:
                continue
            ctl.SetFrameEvent(a.eventId, False)
            pipe = ctl.GetPipelineState()
            ids = []
            for sn, stg in stages:
                sid = rid(pipe.GetShader(stg))
                ids.append(sid)
                if not sid or (sn, sid) in seen:
                    continue
                seen.add((sn, sid))
                refl = pipe.GetShaderReflection(stg)
                rows = ['stage %s id %d entry %s first_eid %d pass %s' % (sn, sid, refl.entryPoint, a.eventId, label)]
                rows.append('-- input signature')
                rows += fmt_sig(refl.inputSignature)
                rows.append('-- output signature')
                rows += fmt_sig(refl.outputSignature)
                for cb in refl.constantBlocks:
                    rows.append('-- cbuffer %s slot %s bytes %d' % (
                        cb.name, attr(cb, 'fixedBindNumber', 'bindPoint'), cb.byteSize))
                    rows += fmt_vars(cb.variables, 0, '    ')
                for r in refl.readOnlyResources:
                    rows.append('-- srv %s slot %s type %s' % (
                        r.name, attr(r, 'fixedBindNumber', 'bindPoint'), str(r.textureType).replace('TextureType.', '')))
                for s in refl.samplers:
                    rows.append('-- sampler %s slot %s' % (s.name, attr(s, 'fixedBindNumber', 'bindPoint')))
                for r in refl.readWriteResources:
                    rows.append('-- uav %s slot %s' % (r.name, attr(r, 'fixedBindNumber', 'bindPoint')))
                rows.append('-- disassembly')
                rows.append(ctl.DisassembleShader(pipe.GetGraphicsPipelineObject(), refl, target))
                with open(os.path.join(OUT, 'shaders', '%s_%d.txt' % (sn, sid)), 'w') as f:
                    f.write('\n'.join(rows) + '\n')
                with open(os.path.join(OUT, 'shaders', '%s_%d.dxbc' % (sn, sid)), 'wb') as f:
                    f.write(bytes(refl.rawBytes))
            dt.write('%d\t%s\t%d\t%d\t%s\n' % (a.eventId, label, a.numIndices, a.numInstances,
                                               '\t'.join(str(i) for i in ids)))
            n += 1
            if n % 250 == 0:
                L('progress', n, a.eventId)
    L('done', n, 'draws', len(seen), 'shaders')
    ctl.Shutdown()
except Exception:
    L(traceback.format_exc())
log.close()
os._exit(0)
