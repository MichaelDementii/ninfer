#!/usr/bin/env python3
"""Бюджет префилла: агрегат по ядрам за весь прогон, плюс занятость и щели между ядрами."""
import csv, sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else '/root/prof/prefill38_cuda_gpu_trace.csv'


def short(n):
    n = n.replace('void ', '').replace('<unnamed>::', '').replace('(anonymous namespace)::', '')
    cut = len(n)
    for i, ch in enumerate(n):
        if ch in '<(':
            cut = i
            break
    return n[:cut].rstrip(': ').split('::')[-1] or n


lines = open(PATH, newline='').readlines()
st = next(i for i, l in enumerate(lines) if l.startswith('Start (ns)'))
rows = []
for r in csv.DictReader(lines[st:]):
    def gi(k):
        try:
            return int(r.get(k))
        except (TypeError, ValueError):
            return 0
    try:
        s, d = float(r['Start (ns)']), float(r['Duration (ns)'])
    except (TypeError, ValueError):
        continue
    rows.append((s, d, short(r.get('Name') or ''),
                 gi('GrdX') * max(1, gi('GrdY')) * max(1, gi('GrdZ')),
                 gi('BlkX') * max(1, gi('BlkY')) * max(1, gi('BlkZ')),
                 gi('Reg/Trd')))
rows.sort()

# Префилл — самый плотный участок трассы: берём отрезок от первого до последнего prompt-ядра.
idx = [i for i, r in enumerate(rows) if 'prompt' in r[2] or 'prefill' in r[2]]
if not idx:
    idx = [0, len(rows) - 1]
lo, hi = min(idx), max(idx)
seg = rows[lo:hi + 1]
span = seg[-1][0] + seg[-1][1] - seg[0][0]
busy = sum(d for _, d, *_ in seg)

agg = defaultdict(lambda: [0, 0.0, 0, 0, 0])
for s, d, n, g, blk, reg in seg:
    a = agg[(n, g)]
    a[0] += 1
    a[1] += d
    a[2] = max(a[2], g)
    a[3] = max(a[3], blk)
    a[4] = max(a[4], reg)

print(f'отрезок префилла: {span/1e6:.1f} мс, ядер {len(seg)}, сумма ядер {busy/1e6:.1f} мс, '
      f'занятость {100*busy/span:.1f}%')
gaps = []
for a, b in zip(seg[:-1], seg[1:]):
    gaps.append(max(0.0, b[0] - (a[0] + a[1])))
print(f'щели между ядрами: сумма {sum(gaps)/1e6:.2f} мс ({100*sum(gaps)/span:.2f}% отрезка), '
      f'медиана {sorted(gaps)[len(gaps)//2]/1e3:.2f} мкс')
print()
print(f'{"ядро":<40}{"сетка":>8}{"мс":>9}{"доля":>7}{"n":>6}{"мкс/зап":>10}{"сетка":>9}{"нитей":>7}{"рег":>5}')
tot = 0.0
for (n, g), (c, t, _g, blk, reg) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:26]:
    tot += t
    print(f'{n[:40]:<40}{g:>8}{t/1e6:>9.1f}{100*t/span:>7.1f}{c:>6}{t/c/1e3:>10.1f}{blk:>7}{reg:>5}')
print(f'{"— показано":<44}{tot/1e6:>9.1f}{100*tot/span:>7.1f}')
