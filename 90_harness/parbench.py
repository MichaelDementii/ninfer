#!/usr/bin/env python3
"""Nagruzka na ninfer-serve: N odnovremennyh zaprosov, kazhdyy so svoim prefiksom.

Prefiks u kazhdogo zaprosa unikalen s pervogo tokena, poetomu obshchiy prefiks ne
pereispolzuetsya dazhe kogda kesh vklyuchen -- inache zamer konkurentnosti merial by kesh.

Imya modeli sprashivaetsya u servera: na proizvolnoe imya on otvechaet 404 model_not_found.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:18080"

BASE = ("The rotary position embedding rotates query and key vectors by an angle that grows "
        "linearly with the token index, so that the dot product between a query and a key "
        "depends only on the distance between their positions and not on where the pair sits "
        "in the sequence. This is why the same attention pattern generalizes when the context "
        "is extended, and why interpolating the rotation base lets a model read further than "
        "it was trained on without retraining every layer from scratch. ")


def make_prompt(i, reps):
    marker = "".join(chr(97 + (i * 7 + k) % 26) for k in range(32))
    return "Session %d marker %s. Read the following note and then summarize it.\n\n%s" % (
        i, marker, BASE * reps)


def discover_model():
    try:
        with urllib.request.urlopen(HOST + "/v1/models", timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = body.get("data") or []
        if data and isinstance(data[0], dict) and data[0].get("id"):
            return data[0]["id"]
    except Exception:  # noqa: BLE001
        pass
    return "ninfer"


def main():
    n_req, reps, max_tokens, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    model_id = discover_model()
    results = [None] * n_req

    def one(i):
        payload = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": make_prompt(i, reps)}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(HOST + "/v1/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            dt = time.perf_counter() - t0
            u = body.get("usage", {})
            results[i] = (dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), None)
        except urllib.error.HTTPError as e:
            results[i] = (time.perf_counter() - t0, 0, 0,
                          "HTTP %s %s" % (e.code, e.read()[:160]))
        except Exception as e:  # noqa: BLE001
            results[i] = (time.perf_counter() - t0, 0, 0, repr(e)[:160])

    t0 = time.perf_counter()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(n_req)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r and r[3] is None]
    bad = [r for r in results if r and r[3] is not None]
    pt = sum(r[1] for r in ok)
    ct = sum(r[2] for r in ok)
    lat = sorted(r[0] for r in ok)
    print("  %-22s model=%-18s zaprosov=%d ok=%d osh=%d  stena=%7.3f s  prompt_tok=%6d  out_tok=%5d"
          % (tag, model_id, n_req, len(ok), len(bad), wall, pt, ct))
    if ok:
        print("      propusk: %8.1f prompt_tok/s  %7.1f out_tok/s   latentnost med=%6.3f max=%6.3f"
              % (pt / wall, ct / wall, lat[len(lat) // 2], lat[-1]))
    for r in bad[:2]:
        print("      OSHIBKA: %s" % r[3])


if __name__ == "__main__":
    main()
