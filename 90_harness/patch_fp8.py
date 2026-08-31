#!/usr/bin/env python3
"""Г7: свип расписаний fp8 small-T для двух геометрий, где лежит больше всего времени.

`Fp8Residual6144` (выход GDN/внимания, 5120x6144, 64 запуска, 8.6% раунда) держит худший КПД
полосы из всех больших GEMM — 77.6% против 86.3% у `Fp8Residual17408` и 85.0% у gate_up. Она же
единственная геометрия, которая при T=4 идёт `TokenPacked`, тогда как AttnInput, GdnInput и
MlpGateUp при T<=3..4 переключаются на `SharedPhase`. У неё K = 6144, то есть 12 фаз — как у
gate_up (10), а не 34 как у Residual17408, где стейджинг проигрывает на барьерах.

Точки: <геометрия>_<что меняем>. Сброс — `base`."""
import subprocess
import sys

R = '/root/ninfer_d4'
P = f'{R}/src/ops/linear/fp8/fp8_config.h'

R6144_OLD = """    using Type = Fp8SmallTSchedule<8, 2, kValuesPerLane, kTokenTile, 1,
                                   Fp8SmallTActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                   kBlockOrder, 1>;"""
GDN_OLD = """    using Type =
        Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                          Fp8CodeCache::Default, 1, Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8MlpGateUpGeometry, ActiveTokens> {"""


def r6144(warps, rows, access, unroll, minb):
    return f"""    using Type = Fp8SmallTSchedule<{warps}, {rows}, kValuesPerLane, kTokenTile, 1,
                                   Fp8SmallTActivationAccess::{access}, Fp8CodeCache::Default,
                                   {unroll}, kBlockOrder, {minb}>;"""


def gdn(warps, rows, access, unroll, minb):
    acc = 'kActivationAccess' if access is None else f'Fp8SmallTActivationAccess::{access}'
    return f"""    using Type =
        Fp8SmallTSchedule<{warps}, {rows}, kValuesPerLane, ActiveTokens, 1, {acc},
                          Fp8CodeCache::Default, {unroll},
                          Fp8SmallTBlockOrder::RowsContiguous, {minb}>;
}};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8MlpGateUpGeometry, ActiveTokens> {{"""


POINTS = {
    'base': [],
    'r6_shared':  [(R6144_OLD, r6144(8, 2, 'SharedPhase', 1, 1))],
    'r6_unr2':    [(R6144_OLD, r6144(8, 2, 'TokenPacked', 2, 1))],
    'r6_w4':      [(R6144_OLD, r6144(4, 2, 'TokenPacked', 1, 1))],
    'r6_w16':     [(R6144_OLD, r6144(16, 2, 'TokenPacked', 1, 1))],
    'r6_rpw4':    [(R6144_OLD, r6144(8, 4, 'TokenPacked', 1, 1))],
    'r6_minb2':   [(R6144_OLD, r6144(8, 2, 'TokenPacked', 1, 2))],
    'gdn_packed': [(GDN_OLD, gdn(8, 2, 'TokenPacked', 1, 1))],
    'gdn_unr2':   [(GDN_OLD, gdn(8, 2, None, 2, 1))],
    'gdn_rpw4':   [(GDN_OLD, gdn(8, 4, None, 1, 1))],
    'gdn_w16':    [(GDN_OLD, gdn(16, 2, None, 1, 1))],
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'base'
    if which not in POINTS:
        sys.exit(f'неизвестная точка {which}')
    subprocess.run(['git', '-C', R, 'checkout', '--', P], check=True)
    text = open(P, encoding='utf-8').read()
    assert len(text) > 6000, 'сброс оставил пустой файл'
    for old, new in POINTS[which]:
        if text.count(old) != 1:
            sys.exit(f'{which}: якорь встречается {text.count(old)} раз')
        text = text.replace(old, new, 1)
    open(P, 'w', encoding='utf-8').write(text)
    print(f'точка {which} наложена')


main()
