"""Every API call of a RenderDoc capture in event ranges, with its parameters (one line per call).

Run inside RenderDoc's embedded Python, like rdc_dump.py:

    set RDC_CAP=D:\\path\\capture.rdc
    set RDC_OUTDIR=D:\\path\\out
    set RDC_RANGES=107700-108000;147030-147310
    qrenderdoc.exe --python tools\\re\\renderdoc\\rdc_events.py

Writes events.txt ("<eventId> <call>(<param>=<value>, ...)", resource ids as numbers, arrays shortened) and
events_log.txt. The script closes qrenderdoc when done.
"""
import os
import re
import traceback

OUT = os.environ.get('RDC_OUTDIR', '.')
CAP = os.environ['RDC_CAP']
RANGES = [tuple(int(x) for x in r.split('-')) for r in os.environ['RDC_RANGES'].split(';') if r.strip()]
log = open(os.path.join(OUT, 'events_log.txt'), 'w')


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


def value(o, depth=0):
    import renderdoc as rd
    B = rd.SDBasic
    try:
        b = o.type.basetype
        if b in (B.Struct, B.Array, B.Chunk):
            n = o.NumChildren()
            parts = []
            for i in range(min(n, 8)):
                c = o.GetChild(i)
                v = value(c, depth + 1) if depth < 3 else '..'
                parts.append(v if b == B.Array else '%s=%s' % (c.name, v))
            return '{%s%s}' % (', '.join(parts), ', ..' if n > 8 else '')
        if b == B.Resource:
            return str(rid(o.AsResourceId()))
        if b == B.Null:
            return 'null'
        if b == B.Float:
            return '%g' % o.AsFloat()
        if b in (B.SignedInteger, B.UnsignedInteger, B.Boolean):
            return str(o.AsInt())
        return o.AsString()
    except Exception as e:
        return '?'


try:
    import renderdoc as rd
    cap = rd.OpenCaptureFile()
    L('open', cap.OpenFile(CAP, '', None))
    st, ctl = cap.OpenCapture(rd.ReplayOptions(), None)
    L('status', st)
    sdf = ctl.GetStructuredFile()
    with open(os.path.join(OUT, 'events.txt'), 'w') as out:
        for a in walk(ctl.GetRootActions()):
            for ev in a.events:
                if not any(lo <= ev.eventId <= hi for lo, hi in RANGES):
                    continue
                ch = sdf.chunks[ev.chunkIndex]
                params = ', '.join('%s=%s' % (ch.GetChild(i).name, value(ch.GetChild(i))) for i in range(ch.NumChildren()))
                out.write('%d %s(%s)\n' % (ev.eventId, ch.name, params))
    L('done')
    ctl.Shutdown()
except Exception:
    L(traceback.format_exc())
log.close()
os._exit(0)
