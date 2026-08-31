#!/usr/bin/env python3
"""Г16: параллелизм ядра recurrent_record.

Ядро читает состояние GDN слоя (3.1 МБ) за 6.66 мкс = 466 ГБ/с при сетке 384 CTA по 128 нитей,
то есть 49 152 нити на машину, держащую 348 160. Оно латентное, а не полосное. Разбиение по
измерению dv независимо: сумма по dk считается внутри столбца, поэтому столбцы делятся свободно.
kDvPerWarp задаёт, сколько столбцов dv несёт один варп, kNumWarps — сколько варпов в CTA.

Аргументы: kDvPerWarp kNumWarps (сброс — 4 4)."""
import subprocess
import sys

R = '/root/ninfer_d4'
P = f'{R}/src/ops/linear_attention/gated_delta_net/recurrent.cuh'

subprocess.run(['git', '-C', R, 'checkout', '--', P], check=True)
s = open(P, encoding='utf-8').read()
assert len(s) > 8000, 'сброс оставил пустой файл'

dv = int(sys.argv[1]) if len(sys.argv) > 1 else 4
nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
if (dv, nw) != (4, 4):
    assert 128 % (dv * nw) == 0, f'kStateDim не делится на kBlockDv={dv*nw}'
    s = s.replace('inline constexpr int kDvPerWarp = 4;',
                  f'inline constexpr int kDvPerWarp = {dv};', 1)
    s = s.replace('inline constexpr int kNumWarps  = 4;',
                  f'inline constexpr int kNumWarps  = {nw};', 1)
    open(P, 'w', encoding='utf-8').write(s)
print(f'kDvPerWarp={dv} kNumWarps={nw} kBlockDv={dv*nw}')
