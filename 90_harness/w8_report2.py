"""Every figure the W8 row-split submission quotes, computed from the two-arm campaign."""
import csv, glob, os, re, statistics, collections

OUT = "/root/qual/gW"
rows = lambda p: list(csv.DictReader(open(p, newline="")))
keyed = lambda p, k: {tuple(r[c] for c in k): float(r["median_us"]) for r in rows(p)}


def moe_table(arm):
    out = {}
    for tk in (1024, 4096, 8192):
        a = {r["codec"]: float(r["median_us"]) for r in rows(f"{OUT}/smoe_mst_{tk}.csv")}
        b = {r["codec"]: float(r["median_us"]) for r in rows(f"{OUT}/smoe_{arm}_{tk}.csv")}
        for c in a:
            out.setdefault(c, {})[tk] = (a[c], b[c], a[c] / b[c])
    return out


print("=== routed MoE prefill, w8 arm (must be flat: this branch does not touch those kernels) ===")
for c, v in sorted(moe_table("w8").items()):
    print("  %-8s " % c + "  ".join("%d: %8.1f -> %8.1f x%.4f" % (t, *v[t]) for t in (1024, 4096, 8192)))


def route_of(path):
    if any(t in path for t in (".mma.r3", ".mma.r4", ".mma.r6")):
        return "w8_rowsplit_gemm_mma"
    if ".mma." in path:
        return "w8_small_t_mma (splitk8)"
    return path.split(".")[2]


a = keyed(f"{OUT}/add_mst.csv", ("path", "T"))
b = keyed(f"{OUT}/add_w8.csv", ("path", "T"))
g = collections.defaultdict(list)
for k in a:
    if k in b:
        g[route_of(k[0])].append((int(k[1]), a[k] / b[k]))
print("\n=== w8_linear_add --production-only, by kernel ===")
for r, v in sorted(g.items()):
    s = sorted(x for _, x in v); ts = sorted(t for t, _ in v)
    print("  %-26s n=%2d T=%d..%-5d median x%.3f best x%.3f worst x%.3f" %
          (r, len(s), ts[0], ts[-1], statistics.median(s), s[-1], s[0]))
    if len(s) < 4:
        print("       rows:", sorted(v))

for name, f, keys in (("w8 linear_swiglu", "swig", ("path", "T")),
                      ("linear suite", "lin", ("label", "qtype", "policy", "N", "K", "T"))):
    A = keyed(f"{OUT}/{f}_mst.csv", keys); B = keyed(f"{OUT}/{f}_w8.csv", keys)
    ks = [k for k in A if k in B]
    mv = [(k, A[k] / B[k]) for k in ks if abs(A[k] / B[k] - 1) > 0.01]
    sp = sorted(s for _, s in mv)
    print(f"\n=== {name}: {len(mv)} of {len(ks)} move" +
          (f"; median x{statistics.median(sp):.3f} best x{sp[-1]:.3f} worst x{sp[0]:.3f}" if mv else ""))
    if f == "lin" and mv:
        print("   non-W8 movers:", [(k[0], k[1], k[5], round(s, 3)) for k, s in mv if k[1] != "W8"])
        print("   W8 movers below 1:", [(k[0], k[5], round(s, 3)) for k, s in mv if k[1] == "W8" and s < 1])
        print("   W8 rows flat:", sum(1 for k in ks if k[1] == "W8" and abs(A[k] / B[k] - 1) <= 0.01),
              "of", sum(1 for k in ks if k[1] == "W8"))
    if f == "swig" and mv:
        print("   movers below 1:", [(k[0], k[1], round(s, 3)) for k, s in mv if s < 1])

for name in ("gdnip", "attnip"):
    A = keyed(f"{OUT}/{name}_mst.csv", ("entry", "format", "policy", "T"))
    B = keyed(f"{OUT}/{name}_w8.csv", ("entry", "format", "policy", "T"))
    print(f"\n=== {name} ===")
    for fm in sorted({k[1] for k in A}):
        cells = sorted((int(k[3]), A[k] / B[k]) for k in A if k[1] == fm and k in B)
        print("   %-9s " % fm + " ".join("%d:%.3f" % (t, s) for t, s in cells))

pair = lambda arm: {int(t): float(v) for t, v in
                    re.findall(r"T=(\d+)\s+median=\s*([0-9.]+)", open(f"{OUT}/pair_{arm}.txt").read())}
pa, pb = pair("mst"), pair("w8")
print("\n=== linear_pair (graph_replay, cold) ===")
print("   " + " ".join("%d:%.3f" % (t, pa[t] / pb[t]) for t in sorted(pa) if t in pb))

print("\n=== GROUPS == 4 extents, k=5120 (r64x16_c48_k128_a1 route) ===")
for n, lo, hi in ((34816, 41, 48), (248320, 34, 48)):
    fa, fb = f"{OUT}/g4_{n}_mst.csv", f"{OUT}/g4_{n}_w8.csv"
    if not (os.path.exists(fa) and os.path.exists(fb)):
        continue
    A, B = keyed(fa, ("T",)), keyed(fb, ("T",))
    cells = sorted((int(k[0]), A[k] / B[k]) for k in A if k in B)
    inr = [s for t, s in cells if lo <= t <= hi]
    out = [s for t, s in cells if not lo <= t <= hi]
    print(f"  n={n}  r64x16_c48_k128_a1 at T={lo}..{hi}: n=%d median x%.3f best x%.3f worst x%.3f"
          % (len(inr), statistics.median(inr), max(inr), min(inr)))
    print(f"        other routes in the same sweep:      n=%d median x%.3f best x%.3f worst x%.3f"
          % (len(out), statistics.median(out), max(out), min(out)))
    print("        " + " ".join("%d:%.3f" % c for c in cells))

print("\n=== repeatability: same arm against itself, three passes ===")
for f, keys in (("add", ("path", "T")), ("swig", ("path", "T"))):
    for arm in ("mst", "w8"):
        ps = [f"{OUT}/rep_{f}_{arm}_{p}.csv" for p in (1, 2, 3)]
        if not all(os.path.exists(p) for p in ps):
            continue
        D = [keyed(p, keys) for p in ps]
        ks = [k for k in D[0] if all(k in d for d in D)]
        rat = [(k, max(d[k] for d in D) / min(d[k] for d in D)) for k in ks]
        rat.sort(key=lambda x: -x[1])
        sp = sorted(r for _, r in rat)
        print("  %-5s %-4s n=%d  median spread x%.3f  p90 x%.3f  worst x%.3f  at %s T=%s"
              % (f, arm, len(sp), statistics.median(sp), sp[int(len(sp) * 0.9)], sp[-1],
                 rat[0][0][0], rat[0][0][1]))
        print("        worst five:", [(k[0].split(".")[-1], k[1], round(r, 3)) for k, r in rat[:5]])

print("\n=== end to end ===")
pts = {}
for p in sorted(glob.glob(os.path.join(OUT, "*.err"))):
    m = re.match(r"(.+)_(MST|W8)_r(\d+)$", os.path.basename(p)[:-4])
    if not m:
        continue
    t = open(p, errors="ignore").read()
    pre = re.findall(r"prefill speed\s+([0-9.]+)", t)
    dec = re.findall(r"decode speed\s+([0-9.]+)", t)
    tok = re.findall(r"prompt tokens\s+(\d+)", t)
    e = pts.setdefault(m.group(1), {"MST": [], "W8": [], "tok": None})
    if pre:
        e[m.group(2)].append((float(pre[-1]), float(dec[-1]) if dec else 0.0))
    if tok:
        e["tok"] = int(tok[0])
for tag in ("c8192_niah_4k", "c8192_niah_16k", "c8192_niah_32k", "c1024_niah_16k", "d27"):
    e = pts.get(tag)
    if not e or not e["MST"] or not e["W8"]:
        continue
    base = statistics.mean(x for x, _ in e["MST"])
    v = statistics.mean(x for x, _ in e["W8"])
    d = statistics.mean(y for _, y in e["W8"]) / statistics.mean(y for _, y in e["MST"]) - 1
    spread = max((max(x for x, _ in e[a]) - min(x for x, _ in e[a])) / statistics.mean(x for x, _ in e[a])
                 for a in ("MST", "W8")) * 100
    print("  %-17s tok=%-6s n=%d  master %8.1f  W8 %+.2f%% (decode %+.2f%%)  spread %.2f%%"
          % (tag, e["tok"], len(e["MST"]), base, (v / base - 1) * 100, d * 100, spread))
    print("      MST " + " ".join("%.1f" % x for x, _ in e["MST"])
          + " | W8 " + " ".join("%.1f" % x for x, _ in e["W8"]))

if os.path.exists(f"{OUT}/sass_w8.txt"):
    print("\n=== sass and resources ===")
    txt = open(f"{OUT}/sass_w8.txt").read()
    print("  " + " | ".join(txt.splitlines()[:2]))
    d = sorted(float(x) for x in re.findall(r"\(([-+][0-9.]+)%\)", txt))
    if d:
        print("  instruction deltas: n=%d %.1f%%..%.1f%% median %.1f%%"
              % (len(d), d[0], d[-1], statistics.median(d)))
    print("  " + "\n  ".join(l for l in open(f"{OUT}/resources_w8.txt").read().splitlines() if l.strip())[:1200])
