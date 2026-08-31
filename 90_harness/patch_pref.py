#!/usr/bin/env python3
"""П1: расписание fp8-MMA префилла.

`fp8_mma_kernel` — 36.4% времени префилла и идёт на 365–379 TFLOP/s при измеренном потолке
515.5 TFLOP/s (микробенч `mmapeak`, воспроизводящий прежние 258 TFLOP/s bf16 и 1026 TOPS s8).
Это 71–74%, тогда как соседний `nvfp4_w4a4_tma_kernel` берёт ~90%. Все пять геометрий при этом
делят одно расписание <64,128,128,2,4,2,2,cg,cg,PingPong,TokenFast> — то есть это умолчание,
а не подбор по формам, в отличие от подобранных декодных small-T расписаний.

Общий объём shared = Stages * (BlockTokens + BlockRows) * BlockK, и он умножается на
MinBlocksPerSm; на sm_120 доступно около 100 КиБ на SM, поэтому глубокие конвейеры идут
с MinBlocksPerSm = 1."""
import subprocess
import sys

R = '/root/ninfer_d4'
P = f'{R}/src/ops/linear/fp8/fp8_a8_schedule.cuh'
OLD = ("Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,\n"
       "                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>")


def sched(bt, br, bk, wt, wr, st, mb, pipe='PingPong', raster='TokenFast', group=None):
    tail = f', {group}' if group is not None else ''
    return (f"Fp8MmaSchedule<{bt}, {br}, {bk}, {wt}, {wr}, {st}, {mb}, Cache::cg, Cache::cg,\n"
            f"                                Fp8MmaFragmentPipeline::{pipe}, "
            f"Fp8MmaRaster::{raster}{tail}>")


POINTS = {
    'base':     None,
    'bt128':    sched(128, 128, 128, 2, 4, 2, 1),
    'bt128w16': sched(128, 128, 128, 4, 4, 2, 1),
    'st3':      sched(64, 128, 128, 2, 4, 3, 1),
    'st4':      sched(64, 128, 128, 2, 4, 4, 1),
    'br256':    sched(64, 256, 128, 2, 4, 2, 1),
    'bk256':    sched(64, 128, 256, 2, 4, 2, 1),
    'bk64':     sched(64, 128, 64, 2, 4, 2, 2),
    'rowfast':  sched(64, 128, 128, 2, 4, 2, 2, raster='RowFast'),
    'grouped':  sched(64, 128, 128, 2, 4, 2, 2, raster='Grouped', group=8),
    'serial':   sched(64, 128, 128, 2, 4, 2, 2, pipe='Serial'),
    'bt128st3': sched(128, 128, 128, 2, 4, 3, 1),
    # Второй заход: держим MinBlocksPerSm = 2 (именно занятость решает) и укладываемся
    # в те же ~48 КиБ shared, но берём тайл с вдвое большей арифметической интенсивностью.
    'bt128k64':   sched(128, 128, 64, 2, 4, 2, 2),
    'bt128k64s3': sched(128, 128, 64, 2, 4, 3, 2),
    'k64s4':      sched(64, 128, 64, 2, 4, 4, 2),
    'bt128br256': sched(128, 256, 64, 2, 8, 2, 2),
    'k64mb4':     sched(64, 128, 64, 2, 4, 2, 4),
    'br256k64':   sched(64, 256, 64, 2, 8, 2, 2),
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'base'
    if which not in POINTS:
        sys.exit(f'неизвестная точка {which}')
    subprocess.run(['git', '-C', R, 'checkout', '--', P], check=True)
    text = open(P, encoding='utf-8').read()
    assert len(text) > 800, 'сброс оставил пустой файл'
    if POINTS[which] is not None:
        n = text.count(OLD)
        if n != 5:
            sys.exit(f'{which}: якорь встречается {n} раз, ожидалось 5')
        text = text.replace(OLD, POINTS[which])
        open(P, 'w', encoding='utf-8').write(text)
    print(f'точка {which} наложена')


main()
