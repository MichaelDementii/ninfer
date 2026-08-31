#!/usr/bin/env python3
"""Бюджет раунда Qwen3.8-27B nvfp4. Формы весов — из bindings.cpp, профиль nvfp4.
Ядра с одинаковым именем и сеткой, но разной формой весов, разделяются по длительности."""
import csv, statistics
from collections import defaultdict
PEAK = 1792e9
H, I, V = 5120, 17408, 248320
def nv(r, c): return r * c * 0.5625     # 4 бита + масштаб fp8 на 16
def b8(r, c): return r * c              # 1 байт

# (имя, CTA, порог мкс) -> (описание, байты). Порог разделяет разные формы под одним ядром.
RULES = [
    ('nvfp4_linear_swiglu_small_t_kernel', 2176, None, 'MLP gate_up  34816x5120  nvfp4', nv(2*I, H)),
    ('nvfp4_small_t_kernel',                640, None, 'MLP down      5120x17408 nvfp4', nv(H, I)),
    ('fp8_a16_small_t_mma_kernel',        15520, None, 'LM-голова   248320x5120  fp8',  b8(V, H)),
    ('fp8_small_t_kernel',                 1024, None, 'GDN qkvz     16384x5120  fp8',  b8(16384, H)),
    ('fp8_small_t_kernel',                  896, None, 'внимание qkgv 14336x5120 fp8',  b8(14336, H)),
    ('fp8_small_t_kernel',                  640, 40.0, 'MLP down      5120x17408 fp8',  b8(H, I)),
    ('fp8_small_t_kernel',                  640,   -1, 'выход GDN/внимания 5120x6144 fp8', b8(H, 6144)),
    ('fp8_mma_kernel',                      272, None, 'MLP gate_up  34816x5120  fp8',  b8(2*I, H)),
    ('w8_small_t_mma_kernel',              2176, None, 'MTP MLP gate_up 34816x5120 w8', b8(2*I, H)),
    ('w8_small_t_mma_kernel',               896, None, 'MTP внимание qkgv 14336x5120 w8', b8(14336, H)),
    ('w8_small_t_mma_kernel',               320, 45.0, 'MTP MLP down  5120x17408 w8',  b8(H, I)),
    ('w8_small_t_mma_kernel',               320, 30.0, 'MTP fc/вход   5120x10240 w8',  b8(H, 2*H)),
    ('w8_small_t_mma_kernel',               320,   -1, 'MTP выход внимания 5120x6144 w8', b8(H, 6144)),
]
def short(n):
    n = n.replace('void ', '').replace('<unnamed>::', '').replace('(anonymous namespace)::', '')
    cut = len(n)
    for i, ch in enumerate(n):
        if ch in '<(':
            cut = i; break
    return n[:cut].rstrip(': ').split('::')[-1] or n
def match(n, g, d_us):
    for kn, kg, thr, desc, by in RULES:
        if n == kn and g == kg and (thr is None or d_us >= thr):
            return desc, by
    return None, None

lines = open('/root/prof/round_d2_cuda_gpu_trace.csv', newline='').readlines()
st = next(i for i, l in enumerate(lines) if l.startswith('Start (ns)'))
rows = []
for r in csv.DictReader(lines[st:]):
    def gi(k):
        try: return int(r.get(k))
        except (TypeError, ValueError): return 0
    try: s, d = float(r['Start (ns)']), float(r['Duration (ns)'])
    except (TypeError, ValueError): continue
    rows.append((s, d, short(r.get('Name') or ''), gi('GrdX')*max(1,gi('GrdY'))*max(1,gi('GrdZ'))))
rows.sort()
lastp = max(i for i, r in enumerate(rows) if 'prompt_i8' in r[2])
dec = rows[lastp+1:]
b = [i for i, r in enumerate(dec) if 'speculative_prepare_verify_inputs' in r[2]]
segs = []
for a, c in zip(b[:-1], b[1:]):
    seg = dec[a:c]; span = seg[-1][0]+seg[-1][1]-seg[0][0]
    if span > 40e6: continue
    segs.append((seg, span))
segs = segs[1:-1]
sp = statistics.median(s for _, s in segs)
med = min(segs, key=lambda x: abs(x[1]-sp))[0]
packs = [i for i, r in enumerate(med) if r[2] == 'mtp_pack_fc_input_kernel']
d0 = packs[0]
while d0 > 0 and med[d0][2] != 'embed_gather_fp8_kernel':
    d0 -= 1

agg = defaultdict(lambda: [0, 0.0, 0.0])
small = defaultdict(float)
for i, (s, d, n, g) in enumerate(med):
    desc, by = match(n, g, d/1e3)
    if desc:
        a = agg[desc]; a[0] += 1; a[1] += d; a[2] += by
    else:
        small[n] += d

print(f'РАУНД {sp/1e3:.1f} мкс   |   паспортная полоса RTX 5090 = {PEAK/1e12:.3f} ТБ/с')
print(f'граница фаз: верификация 0..{d0}, черновик {d0}..{len(med)}\n')
print(f'{"что читает ядро":<34}{"n":>3}{"мкс":>9}{"мкс/шт":>8}{"МБ/шт":>8}{"ГБ/с":>7}{"% полосы":>9}{"% раунда":>9}')
BT = BB = 0.0
for desc, (c, t, by) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    gbs = by / t
    BT += t; BB += by
    print(f'{desc:<34}{c:>3}{t/1e3:>9.1f}{t/c/1e3:>8.2f}{by/c/1e6:>8.1f}{gbs:>7.0f}'
          f'{100*gbs*1e9/PEAK:>9.1f}{100*t/sp:>9.1f}')
print(f'\n{"ИТОГО большие GEMM":<34}{sum(v[0] for v in agg.values()):>3}{BT/1e3:>9.1f}'
      f'{"":>8}{BB/1e6:>8.0f}{BB/BT:>7.0f}{100*BB/BT*1e9/PEAK:>9.1f}{100*BT/sp:>9.1f}')
ST = sum(small.values())
print(f'{"ИТОГО мелкие ядра":<34}{sum(1 for _ in ()):>3}{ST/1e3:>9.1f}{"":>8}{"":>8}{"":>7}{"":>9}{100*ST/sp:>9.1f}')
print(f'\nмелочь по видам (мкс, % раунда):')
for n, t in sorted(small.items(), key=lambda kv: -kv[1])[:14]:
    print(f'  {n[:48]:<48}{t/1e3:>8.1f}{100*t/sp:>8.2f}')
