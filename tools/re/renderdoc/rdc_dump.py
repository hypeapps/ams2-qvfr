"""Dump textures and per-action bindings of a RenderDoc capture.

Run inside RenderDoc's embedded Python (the stock qrenderdoc build has no standalone module):

    set RDC_CAP=D:\\path\\capture.rdc
    set RDC_OUTDIR=D:\\path\\out
    qrenderdoc.exe --python tools\\re\\renderdoc\\rdc_dump.py

Writes to RDC_OUTDIR:
- textures.tsv: every texture (id, name, size, array, mips, MSAA, format, flags, bytes);
- actions.tsv: every draw, dispatch, clear, copy, resolve and present with its outputs, depth target,
  viewport, scissor, read-only resources (VS/PS/CS), UAVs and copy source/destination;
- dump_log.txt: progress and errors.

The script closes qrenderdoc when done. Shader constant values are not dumped: AMS2 streams constants
through one large buffer whose contents RenderDoc 1.46 does not keep, so they read back as zeros.
"""
import os
import re
import traceback

OUT = os.environ.get('RDC_OUTDIR', '.')
CAP = os.environ['RDC_CAP']
log = open(os.path.join(OUT, 'dump_log.txt'), 'w')


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


try:
    import renderdoc as rd
    cap = rd.OpenCaptureFile()
    L('open', cap.OpenFile(CAP, '', None))
    st, ctl = cap.OpenCapture(rd.ReplayOptions(), None)
    L('status', st)
    sdf = ctl.GetStructuredFile()

    names = {rid(r.resourceId): r.name for r in ctl.GetResources()}
    with open(os.path.join(OUT, 'textures.tsv'), 'w') as t:
        t.write('id\tname\tw\th\td\tarr\tmips\tms\tfmt\tflags\tbytes\n')
        for x in ctl.GetTextures():
            i = rid(x.resourceId)
            t.write('%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%d\n' % (
                i, names.get(i, ''), x.width, x.height, x.depth, x.arraysize, x.mips, x.msSamp,
                x.format.Name(), str(x.creationFlags), x.byteSize))

    F = rd.ActionFlags
    want = F.Drawcall | F.Dispatch | F.Clear | F.Resolve | F.Copy | F.GenMips | F.Present
    acts = [a for a in walk(ctl.GetRootActions()) if a.flags & want]
    L('actions', len(acts))
    with open(os.path.join(OUT, 'actions.tsv'), 'w') as out:
        out.write('eid\tname\tflags\tidx\tinst\touts\tdepth\tvp\tscissor\tps_srv\tvs_srv\tcs_srv\tuav\tcopy\n')
        for n, a in enumerate(acts, 1):
            outs = ','.join(str(rid(o)) for o in a.outputs if rid(o))
            vp = sc = uav = copy = ''
            srv = {'vs': '', 'ps': '', 'cs': ''}
            if a.flags & (F.Copy | F.Resolve):
                copy = '%d->%d' % (rid(a.copySource), rid(a.copyDestination))
            if a.flags & (F.Drawcall | F.Dispatch):
                ctl.SetFrameEvent(a.eventId, False)
                pipe = ctl.GetPipelineState()
                v = pipe.GetViewport(0)
                vp = '%g,%g,%g,%g' % (v.x, v.y, v.width, v.height)
                s = pipe.GetScissor(0)
                sc = '%g,%g,%g,%g%s' % (s.x, s.y, s.width, s.height, '' if s.enabled else 'off')
                compute = bool(a.flags & F.Dispatch)
                stages = [('cs', rd.ShaderStage.Compute)] if compute else \
                    [('vs', rd.ShaderStage.Vertex), ('ps', rd.ShaderStage.Pixel)]
                for sn, stg in stages:
                    ids = sorted({rid(u.descriptor.resource) for u in pipe.GetReadOnlyResources(stg, True)})
                    srv[sn] = ','.join(str(i) for i in ids if i)
                stg = rd.ShaderStage.Compute if compute else rd.ShaderStage.Pixel
                ids = sorted({rid(u.descriptor.resource) for u in pipe.GetReadWriteResources(stg, True)})
                uav = ','.join(str(i) for i in ids if i)
            out.write('%d\t%s\t%s\t%d\t%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
                a.eventId, a.GetName(sdf).replace('\t', ' '), str(a.flags).replace('ActionFlags.', ''),
                a.numIndices, a.numInstances, outs, rid(a.depthOut), vp, sc, srv['ps'], srv['vs'], srv['cs'],
                uav, copy))
            if n % 500 == 0:
                L('progress', n, a.eventId)
    L('done', len(acts))
    ctl.Shutdown()
except Exception:
    L(traceback.format_exc())
log.close()
os._exit(0)
