#!/usr/bin/env python3
"""Refresh the two comments that still describe the sixteen-group activation-scale tile."""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])


def edit(rel, old, new):
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit("anchor count %d in %s" % (text.count(old), rel))
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("patched", rel)


edit(
    "src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh",
    """    // Activation scales arrive tile-contiguous: one [BlockM tokens, 16 groups] tile is BlockM
    // bytes wide and 16 rows tall, so the request is wide instead of BlockM separate 16-byte ones.
    // A K128 tile consumes the first eight of the sixteen group bytes; the rest is look-ahead.""",
    """    // Activation scales arrive tile-contiguous: one [BlockM tokens, 8 groups] tile is BlockM
    // bytes wide and 8 rows tall, so the request is wide instead of BlockM separate 16-byte ones.
    // Eight groups is exactly what a K128 stage consumes, so no scale byte is fetched twice.""",
)

edit(
    "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu",
    """    // Activation scales arrive tile-contiguous, one [BlockM tokens, 16 groups] tile per request.""",
    """    // Activation scales arrive tile-contiguous, one [BlockM tokens, 8 groups] tile per request:
    // eight groups is exactly what a K128 stage consumes.""",
)

print("done")
