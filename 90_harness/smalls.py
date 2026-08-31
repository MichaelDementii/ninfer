#!/usr/bin/env python3
"""Мелкие ядра медианного раунда: счётчик, сетка, мкс на запуск. Порядок — по суммарному времени."""
import csv, statistics, sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else '/root/prof/round_d2_cuda_gpu_trace.csv'
BIG = ('nvfp4_linear_swiglu_small_t_kernel', 'nvfp4_small_t_kernel', 'fp8_a16_small_t_mma_kernel',
       'fp8_small_t_kernel', 'fp8_mma_kernel', 'w8_small_t_mma_kernel')


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
                 gi('BlkX') * max(1, gi('BlkY')) * max(1, gi('BlkZ'))))
rows.sort()
lastp = max(i for i, r in enumerate(rows) if 'prompt_i8' in r[2])
dec = rows[lastp + 1:]
b = [i for i, r in enumerate(dec) if 'speculative_prepare_verify_inputs' in r[2]]
segs = []
for a, c in zip(b[:-1], b[1:]):
    seg = dec[a:c]
    span = seg[-1][0] + seg[-1][1] - seg[0][0]
    if span > 40e6:
        continue
    segs.append((seg, span))
segs = segs[1:-1]
sp = statistics.median(s for _, s in segs)
med = min(segs, key=lambda x: abs(x[1] - sp))[0]

agg = defaultdict(lambda: [0, 0.0])
for s, d, n, g, blk in med:
    if n in BIG:
        continue
    a = agg[(n, g, blk)]
    a[0] += 1
    a[1] += d
print(f'раунд {sp/1e3:.1f} мкс, всего ядер {len(med)}')
print(f'{"ядро":<46}{"сетка":>7}{"нитей":>7}{"n":>4}{"мкс":>9}{"мкс/шт":>8}{"%раунда":>9}')
tot = 0.0
for (n, g, blk), (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    tot += t
    print(f'{n[:46]:<46}{g:>7}{blk:>7}{c:>4}{t/1e3:>9.1f}{t/c/1e3:>8.2f}{100*t/sp:>9.2f}')
print(f'{"ИТОГО мелких":<46}{"":>7}{"":>7}{sum(v[0] for v in agg.values()):>4}'
      f'{tot/1e3:>9.1f}{"":>8}{100*tot/sp:>9.2f}')
