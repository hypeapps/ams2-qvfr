"""Depth test state of the draws in event ranges of a RenderDoc capture (D3D11), grouped into runs.

Run inside RenderDoc's embedded Python, like rdc_dump.py:

    set RDC_CAP=D:\\path\\capture.rdc
    set RDC_OUTDIR=D:\\path\\out
    set RDC_RANGES=24958-48235
    qrenderdoc.exe --python tools\\re\\renderdoc\\rdc_depth_state.py

Writes depth_state.txt: one line per run of consecutive draws with the same depth target, depth enable, function,
write mask and first render target ("<first eid>-<last eid> draws <n> ds <id> enable <0|1> func <name> write <0|1>
rt0 <id>"), then totals per (depth target, function, write). Log in depth_state_log.txt. Closes qrenderdoc when done.
"""
import os
import re
import traceback

OUT = os.environ.get('RDC_OUTDIR', '.')
CAP = os.environ['RDC_CAP']
RANGES = [tuple(int(x) for x in r.split('-')) for r in os.environ['RDC_RANGES'].split(';') if r.strip()]
log = open(os.path.join(OUT, 'depth_state_log.txt'), 'w')


def L(*a):
    log.write(' '.join(str(x) for x in a) + '\n')
    log.flush()


def walk(xs):
    for x in xs:
        yield x
        yield from walk(x.children)


def rid(r):
    m = re.search(r'(\d+)', str(r))
    return int(m.group(1)) if m else 0


try:
    import renderdoc as rd
    cap = rd.OpenCaptureFile()
    L('open', cap.OpenFile(CAP, '', None))
    st, ctl = cap.OpenCapture(rd.ReplayOptions(), None)
    L('status', st)
    draws = [a for a in walk(ctl.GetRootActions())
             if (a.flags & rd.ActionFlags.Drawcall) and any(lo <= a.eventId <= hi for lo, hi in RANGES)]
    L('draws', len(draws))
    runs, totals = [], {}
    for i, a in enumerate(draws):
        ctl.SetFrameEvent(a.eventId, False)
        ps = ctl.GetD3D11PipelineState()
        om = ps.outputMerger
        dss = om.depthStencilState
        key = (rid(a.depthOut), int(dss.depthEnable), str(dss.depthFunction).split('.')[-1],
               int(dss.depthWrites), rid(a.outputs[0]) if len(a.outputs) else 0)
        if runs and runs[-1][3] == key:
            runs[-1][1] = a.eventId
            runs[-1][2] += 1
        else:
            runs.append([a.eventId, a.eventId, 1, key])
        tk = key[:4]
        totals[tk] = totals.get(tk, 0) + 1
        if i % 500 == 0:
            L('progress', i, a.eventId)
    with open(os.path.join(OUT, 'depth_state.txt'), 'w') as out:
        for lo, hi, n, (ds, en, fn, wr, rt0) in runs:
            out.write('%d-%d draws %d ds %d enable %d func %s write %d rt0 %d\n' % (lo, hi, n, ds, en, fn, wr, rt0))
        out.write('\ntotals (ds, enable, func, write): draws\n')
        for k, n in sorted(totals.items(), key=lambda kv: -kv[1]):
            out.write('%s: %d\n' % (k, n))
    L('done')
    ctl.Shutdown()
except Exception:
    L(traceback.format_exc())
log.close()
os._exit(0)
