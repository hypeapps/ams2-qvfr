"""List eye-sized textures from an rdc_dump.py output: who writes and reads them in each frame phase.

    python -I tools/re/renderdoc/eye_targets.py <dump-dir> <eye-w> <eye-h> <eye1-first-eid> <eye1-last-eid> \
        <eye2-first-eid> <eye2-last-eid>

A texture counts as eye-sized when it is the eye size divided by 1, 2, 4 or 8 in both axes.
Usage letters: W render target, D depth target, U UAV, R shader read, cs/cd copy source/destination.
Also lists draws in the eye phases whose viewport does not cover their first output.
"""
import collections
import csv
import os
import sys

d, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
e1a, e1b, e2a, e2b = (int(x) for x in sys.argv[4:8])
tex = {int(r['id']): r for r in csv.DictReader(open(os.path.join(d, 'textures.tsv')), delimiter='\t')}
acts = list(csv.DictReader(open(os.path.join(d, 'actions.tsv')), delimiter='\t'))


def ids(s):
    return [int(x) for x in s.split(',') if x] if s else []


def phase(e):
    if e < e1a:
        return 'frame'
    if e <= e1b:
        return 'eye1'
    if e < e2a:
        return 'between'
    if e <= e2b:
        return 'eye2'
    return 'end'


def eye_div(t):
    for k in (1, 2, 4, 8):
        if int(t['w']) == W // k and int(t['h']) == H // k:
            return k
    return 0


use = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
first = {}
mismatch = collections.Counter()
for a in acts:
    e = int(a['eid'])
    ph = phase(e)

    def mark(i, kind):
        if i in tex:
            use[i][ph][kind] += 1
            first.setdefault(i, (e, kind))

    for i in ids(a['outs']):
        mark(i, 'W')
    if a['depth'] != '0':
        mark(int(a['depth']), 'D')
    for i in ids(a['uav']):
        mark(i, 'U')
    for i in ids(a['ps_srv']) + ids(a['vs_srv']) + ids(a['cs_srv']):
        mark(i, 'R')
    if a['copy']:
        s, t = a['copy'].split('->')
        mark(int(s), 'cs')
        mark(int(t), 'cd')
    if ph in ('eye1', 'eye2') and a['vp']:
        o = ids(a['outs']) or ([int(a['depth'])] if a['depth'] != '0' else [])
        if o and o[0] in tex:
            t = tex[o[0]]
            if [float(v) for v in a['vp'].split(',')] != [0, 0, float(t['w']), float(t['h'])]:
                mismatch[(ph, '%sx%s' % (t['w'], t['h']), a['vp'])] += 1

print('id\tdiv\tformat\tmsaa\tframe\teye1\teye2\tend\tfirst-use')
for i in sorted(use):
    k = eye_div(tex[i])
    if not k:
        continue
    cols = [''.join('%s%d' % (x, c) for x, c in sorted(use[i][ph].items())) or '-'
            for ph in ('frame', 'eye1', 'eye2', 'end')]
    print('\t'.join([str(i), '1/%d' % k, tex[i]['fmt'], tex[i]['ms']] + cols + ['%d %s' % first[i]]))
unused = sorted(i for i in tex if eye_div(tex[i]) and i not in use)
print('unused eye-sized textures (%d): %s' % (len(unused), unused))
print('eye draws with a partial viewport: %d' % sum(mismatch.values()))
for k, v in sorted(mismatch.items()):
    print(' ', k, v)
