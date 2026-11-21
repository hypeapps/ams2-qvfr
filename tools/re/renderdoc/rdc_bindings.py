"""Per-slot bindings of chosen events of a RenderDoc capture (D3D11), plus texture snapshots with value samples.

Run inside RenderDoc's embedded Python, like rdc_dump.py:

    set RDC_CAP=D:\\path\\capture.rdc
    set RDC_OUTDIR=D:\\path\\out
    set RDC_EIDS=32830,32866,36831
    set RDC_SAVE=32865:15415;32866:15415     (optional: <eid>:<texture id>, state after that event)
    set RDC_GRID=8x6                          (optional: value samples per saved texture, default 8x6)
    qrenderdoc.exe --python tools\\re\\renderdoc\\rdc_bindings.py

Writes to RDC_OUTDIR:
- bind_<eid>.txt: render targets (id, name, format, size) with the blend state and write mask of each, depth target
  with depth and stencil state, viewport, and for VS, PS and CS the shader id and every used constant buffer
  (slot, buffer id, offset, size), shader resource (slot, reflection name, texture id, name, view format, mips,
  texture size and format) and sampler (slot, filter, address modes, comparison);
- tex_<eid>_<id>_c<n>.png: one channel of a saved texture, stretched from its minimum to its maximum;
- tex_<eid>_<id>.txt: texture description and creation flags, per-channel minimum and maximum, and the raw values
  on a grid of pixels (sample 0 of MSAA textures);
- bindings_log.txt: progress and errors.

Constant values are not dumped (see rdc_dump.py). The script closes qrenderdoc when done.
"""
import os
import re
import traceback

OUT = os.environ['RDC_OUTDIR']
CAP = os.environ['RDC_CAP']
EIDS = [int(x) for x in os.environ.get('RDC_EIDS', '').split(',') if x.strip()]
SAVE = [tuple(int(v) for v in x.split(':')) for x in os.environ.get('RDC_SAVE', '').split(';') if x.strip()]
GX, GY = (int(v) for v in os.environ.get('RDC_GRID', '8x6').split('x'))
log = open(os.path.join(OUT, 'bindings_log.txt'), 'w')


def L(*a):
    log.write(' '.join(str(x) for x in a) + '\n')
    log.flush()


def rid(r):
    m = re.search(r'(\d+)', str(r))
    return int(m.group(1)) if m else 0


def en(v):
    return str(v).split('.')[-1]


def slot(refl_list, idx):
    if refl_list is None or idx >= len(refl_list):
        return '?', '?'
    r = refl_list[idx]
    return getattr(r, 'fixedBindNumber', getattr(r, 'bindPoint', '?')), r.name


try:
    import renderdoc as rd
    cap = rd.OpenCaptureFile()
    L('open', cap.OpenFile(CAP, '', None))
    st, ctl = cap.OpenCapture(rd.ReplayOptions(), None)
    L('status', st)
    sdf = ctl.GetStructuredFile()
    names = {rid(r.resourceId): r.name for r in ctl.GetResources()}
    texs = {rid(t.resourceId): t for t in ctl.GetTextures()}
    acts = {}
    stack = list(ctl.GetRootActions())
    while stack:
        a = stack.pop()
        acts[a.eventId] = a
        stack.extend(a.children)

    def tdesc(i):
        t = texs.get(i)
        if t is None:
            return '%d %s (not a texture)' % (i, names.get(i, ''))
        return '%d %s %dx%d arr %d mips %d ms %d %s [%s]' % (
            i, names.get(i, ''), t.width, t.height, t.arraysize, t.mips, t.msSamp, t.format.Name(),
            en(t.creationFlags).replace('TextureCategory.', ''))

    def do_bind(e):
        ctl.SetFrameEvent(e, True)
        pipe = ctl.GetPipelineState()
        d3d = ctl.GetD3D11PipelineState()
        a = acts.get(e)
        rows = ['event %d %s' % (e, a.GetName(sdf) if a else '?')]
        v = pipe.GetViewport(0)
        rows.append('viewport %g,%g %gx%g depth %g..%g' % (v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth))
        blends = pipe.GetColorBlends()
        for n, o in enumerate(pipe.GetOutputTargets()):
            if not rid(o.resource):
                continue
            b = blends[n] if n < len(blends) else None
            bs = 'blend off'
            if b is not None and b.enabled:
                bs = 'blend color %s %s %s alpha %s %s %s' % (
                    en(b.colorBlend.source), en(b.colorBlend.destination), en(b.colorBlend.operation),
                    en(b.alphaBlend.source), en(b.alphaBlend.destination), en(b.alphaBlend.operation))
            mask = b.writeMask if b is not None else '?'
            rows.append('rt%d %s view %s mip %d | %s mask 0x%x' % (
                n, tdesc(rid(o.resource)), o.format.Name(), o.firstMip, bs, mask if mask != '?' else 0))
        dt = pipe.GetDepthTarget()
        dss = d3d.outputMerger.depthStencilState
        rows.append('depth %s | test %d %s write %d stencil %d ref %d' % (
            tdesc(rid(dt.resource)) if rid(dt.resource) else 'none', int(dss.depthEnable),
            en(dss.depthFunction), int(dss.depthWrites), int(dss.stencilEnable), dss.frontFace.reference))
        for sn, stg in (('VS', rd.ShaderStage.Vertex), ('PS', rd.ShaderStage.Pixel), ('CS', rd.ShaderStage.Compute)):
            refl = pipe.GetShaderReflection(stg)
            if refl is None:
                continue
            rows.append('== %s %d' % (sn, rid(pipe.GetShader(stg))))
            for i, cb in enumerate(refl.constantBlocks):
                try:
                    u = pipe.GetConstantBlock(stg, i, 0)
                    rows.append('  cb slot %s %s buf %d off %d size %d' % (
                        getattr(cb, 'fixedBindNumber', '?'), cb.name, rid(u.descriptor.resource),
                        u.descriptor.byteOffset, u.descriptor.byteSize))
                except Exception as ex:
                    rows.append('  cb %s ? %s' % (cb.name, ex))
            for u in pipe.GetReadOnlyResources(stg, True):
                bp, nm = slot(refl.readOnlyResources, u.access.index)
                d = u.descriptor
                rows.append('  srv slot %s %-34s -> %s | view %s mip %d+%d' % (
                    bp, nm, tdesc(rid(d.resource)), d.format.Name(), d.firstMip, d.numMips))
            for u in pipe.GetReadWriteResources(stg, True):
                bp, nm = slot(refl.readWriteResources, u.access.index)
                rows.append('  uav slot %s %s -> %s' % (bp, nm, tdesc(rid(u.descriptor.resource))))
            for u in pipe.GetSamplers(stg, True):
                bp, nm = slot(refl.samplers, u.access.index)
                s = u.sampler
                rows.append('  smp slot %s %s filter %s/%s/%s %s addr %s %s %s cmp %s' % (
                    bp, nm, en(s.filter.minify), en(s.filter.magnify), en(s.filter.mip), en(s.filter.filter),
                    en(s.addressU), en(s.addressV), en(s.addressW), en(s.compareFunction)))
        with open(os.path.join(OUT, 'bind_%d.txt' % e), 'w') as f:
            f.write('\n'.join(rows) + '\n')
        L('bind', e)

    def do_save(e, i):
        t = texs.get(i)
        if t is None:
            L('save', e, i, 'not a texture')
            return
        ctl.SetFrameEvent(e, True)
        sub = rd.Subresource(0, 0, 0)
        lo, hi = ctl.GetMinMax(t.resourceId, sub, rd.CompType.Typeless)
        rows = [tdesc(i), 'after event %d' % e]
        nch = t.format.compCount
        rows.append('min ' + ' '.join('%g' % lo.floatValue[c] for c in range(nch)))
        rows.append('max ' + ' '.join('%g' % hi.floatValue[c] for c in range(nch)))
        for c in range(nch):
            s = rd.TextureSave()
            s.resourceId = t.resourceId
            s.destType = rd.FileType.PNG
            s.mip = 0
            s.slice.sliceIndex = 0
            s.sample.sampleIndex = 0
            s.channelExtract = c
            s.alpha = rd.AlphaMapping.Discard
            s.comp.blackPoint = lo.floatValue[c]
            s.comp.whitePoint = hi.floatValue[c] if hi.floatValue[c] > lo.floatValue[c] else lo.floatValue[c] + 1
            ctl.SaveTexture(s, os.path.join(OUT, 'tex_%d_%d_c%d.png' % (e, i, c)))
        rows.append('grid %dx%d: x y -> values' % (GX, GY))
        for gy in range(GY):
            for gx in range(GX):
                x = int((gx + 0.5) * t.width / GX)
                y = int((gy + 0.5) * t.height / GY)
                p = ctl.PickPixel(t.resourceId, x, y, sub, rd.CompType.Typeless)
                rows.append('%5d %5d  %s' % (x, y, ' '.join('%.6g' % p.floatValue[c] for c in range(nch))))
        with open(os.path.join(OUT, 'tex_%d_%d.txt' % (e, i)), 'w') as f:
            f.write('\n'.join(rows) + '\n')
        L('save', e, i)

    for e in EIDS:
        try:
            do_bind(e)
        except Exception:
            L('bind', e, traceback.format_exc())
    for e, i in SAVE:
        try:
            do_save(e, i)
        except Exception:
            L('save', e, i, traceback.format_exc())
    L('done')
    ctl.Shutdown()
except Exception:
    L(traceback.format_exc())
log.close()
os._exit(0)
