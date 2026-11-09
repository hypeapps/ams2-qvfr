"""Every use of chosen resources in a RenderDoc capture (event, kind of use), optionally in event ranges.

Run inside RenderDoc's embedded Python, like rdc_dump.py:

    set RDC_CAP=D:\\path\\capture.rdc
    set RDC_OUTDIR=D:\\path\\out
    set RDC_RES=15454;15462
    set RDC_RANGES=40000-60000        (optional)
    qrenderdoc.exe --python tools\\re\\renderdoc\\rdc_usage.py

Writes usage.txt ("<resource> <eventId> <usage> <action name>") and usage_log.txt. The script closes qrenderdoc when
done.
"""
import os
import re
import traceback

OUT = os.environ.get('RDC_OUTDIR', '.')
CAP = os.environ['RDC_CAP']
RES = [int(x) for x in os.environ['RDC_RES'].split(';') if x.strip()]
RANGES = [tuple(int(x) for x in r.split('-')) for r in os.environ.get('RDC_RANGES', '').split(';') if r.strip()]
log = open(os.path.join(OUT, 'usage_log.txt'), 'w')


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
    names = {}
    for a in walk(ctl.GetRootActions()):
        names[a.eventId] = a.GetName(ctl.GetStructuredFile())
    byId = {rid(r.resourceId): r for r in ctl.GetResources()}
    with open(os.path.join(OUT, 'usage.txt'), 'w') as out:
        for n in RES:
            r = byId.get(n)
            if r is None:
                out.write('%d not found\n' % n)
                continue
            out.write('# %d %s\n' % (n, r.name))
            for u in ctl.GetUsage(r.resourceId):
                if RANGES and not any(lo <= u.eventId <= hi for lo, hi in RANGES):
                    continue
                out.write('%d %d %s %s\n' % (n, u.eventId, str(u.usage), names.get(u.eventId, '')))
    L('done')
    ctl.Shutdown()
except Exception:
    L(traceback.format_exc())
log.close()
os._exit(0)
