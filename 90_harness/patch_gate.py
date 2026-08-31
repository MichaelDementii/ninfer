#!/usr/bin/env python3
"""Г6b: маршрут gating-проекции при T=4.

Сейчас cols 2..8 у 27B идут в SmallTSplit10 — это два ядра, partial (сетка 240, 3.15 мкс) и
reduce (сетка 2, 1.71 мкс), вместе 233 мкс за раунд = 1.42%. Рядом в том же каталоге лежат
одноядерные расписания (MmaUnsplit и кооперативные split-варианты), которые на этих же формах
уже используются для больших T. Проверяем их на T=4: одно ядро вместо двух убирает и запуск,
и межблочную редукцию.
"""
import subprocess
import sys

R = '/root/ninfer_d4'
P = f'{R}/src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp'
OLD = "    {{2, 8}, Bf16GdnGatingScheduleId::SmallTSplit10},"

IDS = ['SmallTSplit10', 'MmaUnsplit', 'MmaCooperativeSplit8', 'MmaCooperativeSplit4',
       'MmaCooperativeSplit2', 'MmaCooperativeSplit16', 'SimtWarpRowC8', 'SimtWarpRowC4']

which = sys.argv[1] if len(sys.argv) > 1 else 'SmallTSplit10'
if which not in IDS:
    sys.exit(f'неизвестное расписание {which}')
subprocess.run(['git', '-C', R, 'checkout', '--', P], check=True)
s = open(P, encoding='utf-8').read()
assert len(s) > 4000, 'сброс оставил пустой файл'
if which != 'SmallTSplit10':
    if s.count(OLD) != 1:
        sys.exit(f'якорь встречается {s.count(OLD)} раз')
    s = s.replace(OLD, f"    {{{{2, 8}}, Bf16GdnGatingScheduleId::{which}}},", 1)
    open(P, 'w', encoding='utf-8').write(s)
print(f'маршрут T=2..8 -> {which}')
