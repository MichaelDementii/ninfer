#!/usr/bin/env python3
"""Г14: переключить путь чтения масштабов nvfp4 у small-T расписаний с Direct на StagedRaw.

Direct читает по одному байту на полосу; из-за раскладки масштабов (тайл 128 строк x 4 группы
= 512 байт) 32 полосы варпа попадают в 8 разных мест с шагом 512 Б — восемь секторов ради
32 байт. StagedRaw уже реализован и используется на пути T=1: CTA загружает свои масштабы
векторами uint4 в shared и раздаёт их через __shfl. Индексация у обоих путей совпадает
байт в байт, так что это чистая замена доступа.

Аргумент — какие геометрии переключать: gen | r17 | both | all
"""
import re
import subprocess
import sys

ROOT = '/root/ninfer_d4'
CFG = f'{ROOT}/src/ops/linear/nvfp4/nvfp4_config.h'

# Каждая специализация опознаётся по строке-заголовку, идущей перед ней.
MARKS = {
    'gen': 'struct Nvfp4LinearSmallTProductionSchedule {',
    'gdn': 'struct Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens> {',
    'r61': 'struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Geometry, ActiveTokens> {',
    'r17': 'struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Geometry, ActiveTokens> {',
}
SETS = {'gen': ['gen'], 'r17': ['r17'], 'both': ['gen', 'r17'],
        'all': ['gen', 'gdn', 'r61', 'r17'], 'base': []}


def reset():
    subprocess.run(['git', '-C', ROOT, 'checkout', '--', CFG], check=True)
    text = open(CFG, encoding='utf-8').read()
    assert len(text) > 4000, 'сброс оставил пустой файл'
    return text


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'base'
    if which not in SETS:
        sys.exit(f'неизвестный набор {which}')
    text = reset()
    for key in SETS[which]:
        start = text.index(MARKS[key]) + len(MARKS[key])
        end = text.index('};', start)
        body = text[start:end]
        assert body.count('Nvfp4ScaleAccess::Direct') == 1, f'{key}: якорь не один'
        text = text[:start] + body.replace('Nvfp4ScaleAccess::Direct',
                                           'Nvfp4ScaleAccess::StagedRaw') + text[end:]
    open(CFG, 'w', encoding='utf-8').write(text)
    got = len(re.findall('Nvfp4ScaleAccess::StagedRaw', text))
    print(f'{which}: StagedRaw теперь в {got} местах (одно из них — путь T=1)')


main()
