#!/usr/bin/env python3
"""Что меняется в раунде декода при переходе с 4K на 32K контекста.
Берём медианный раунд из каждой трассы и сравниваем по видам ядер."""
import csv, statistics, sys
from collections import defaultdict


def short(n):
    n = n.replace('void ', '').replace('<unnamed>::', '').replace('(anonymous namespace)::', '')
    cut = len(n)
    for i, ch in enumerate(n):
        if ch in '<(':
            cut = i
            break
    return n[:cut].rstrip(': ').split('::')[-1] or n


def median_round(path):
    lines = open(path, newline='').readlines()
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
                     gi('GrdX') * max(1, gi('GrdY')) * max(1, gi('GrdZ'))))
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
    return med, sp, len(segs)


def agg(med):
    a = defaultdict(lambda: [0, 0.0])
    for s, d, n, g in med:
        k = a[(n, g)]
        k[0] += 1
        k[1] += d
    return a


m4, sp4, n4 = median_round('/root/prof/round_38nv_cuda_gpu_trace.csv')
m32, sp32, n32 = median_round('/root/prof/round_38nv_32k_cuda_gpu_trace.csv')
a4, a32 = agg(m4), agg(m32)

print(f'раунд 4K  = {sp4/1e3:9.1f} мкс, ядер {len(m4):4d}, раундов в трассе {n4}')
print(f'раунд 32K = {sp32/1e3:9.1f} мкс, ядер {len(m32):4d}, раундов в трассе {n32}')
print(f'разница   = {(sp32-sp4)/1e3:9.1f} мкс  ({100*(sp32-sp4)/sp4:+.1f}%)\n')

keys = set(a4) | set(a32)
rows = []
for k in keys:
    c4, t4 = a4.get(k, (0, 0.0))
    c32, t32 = a32.get(k, (0, 0.0))
    rows.append((t32 - t4, k, c4, t4, c32, t32))
rows.sort(reverse=True)
print(f'{"ядро":<46}{"CTA":>7}{"n":>4}{"4K мкс":>9}{"32K мкс":>9}{"Δмкс":>9}{"Δ%раунда":>10}')
for d, (n, g), c4, t4, c32, t32 in rows:
    if abs(d) < 3e3 and abs(t32) < 200e3:
        continue
    print(f'{n[:46]:<46}{g:>7}{c32:>4}{t4/1e3:>9.1f}{t32/1e3:>9.1f}{d/1e3:>9.1f}'
          f'{100*d/sp4:>10.2f}')
tail = sum(d for d, k, *_ in rows if abs(d) < 3e3 and abs(k[0] and 0) == 0)
print(f'\nсумма Δ по всем ядрам: {sum(r[0] for r in rows)/1e3:.1f} мкс')
