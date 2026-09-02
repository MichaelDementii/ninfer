#!/usr/bin/env python3
"""Konfigi ocenki kachestva dlya 35B-A3B pod nashu mashinu.

Vyborka -- avtorskaya iz qwen3_6_35b_{aime,gpqa}.yaml. Byudzhety vyvoda i
konkurentnost -- avtorskie iz qwen3_8_27b_groupwise_reasoning.yaml (oni uzhe
dokazany na etoy karte). Puti NIAH perepisany s /home/neroued na lokalnyy kesh.
Odna i ta zhe vyborka na obeih rukah: menyaetsya tolko --kv-dtype servera.
"""
import pathlib, sys

CTX = 262144
if "--ctx" in sys.argv:
    CTX = int(sys.argv[sys.argv.index("--ctx") + 1])
# Byudzhet vyvoda ne dolzhen prevyshat kontekst za vychetom prompta i zapasa.
AIME_MAX = min(65536, CTX - 8192)
IFB_MAX  = min(65536, CTX - 8192)
GPQA_MAX = min(65536, CTX - 8192)

OUT = pathlib.Path("/root/exp/kv/cfg")
OUT.mkdir(parents=True, exist_ok=True)

CORPUS = "/root/.cache/modelscope/datasets/AI-ModelScope--Needle-in-a-Haystack-Corpus/snapshots/master"
TOKENIZER = "/root/.cache/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"

TARGET = """schema_version: 1

targets:
  qwen35_api:
    protocol: openai_chat
    base_url: http://127.0.0.1:18080/v1
    model: qwen3.6-35b-a3b
    max_concurrency: {conc}
    request:
      timeout_seconds: 86400
      retries: 2
      retry_interval_seconds: 10
"""

RUNTIME = """
runtime:
  max_parallel_jobs: 1
  runs_dir: eval/runs
  progress:
    enabled: true
    refresh_seconds: 1
    heartbeat_seconds: 900
  samples:
    retention: all
"""

# Vyborka 35B, slovo v slovo iz konfigov avtora dlya etoy modeli.
GEN = """        generation:
          temperature: 0.6
          top_p: 0.95
          top_k: 20
          presence_penalty: 1.0
          frequency_penalty: 0.0
          max_tokens: {max_tokens}
          seed: 42
"""

def job(jid, dataset, conc, max_tokens, few_shot=True):
    s = f"""      - id: {jid}
        backend: evalscope
        dataset: {dataset}
        target: qwen35_api
        max_concurrency: {conc}
"""
    s += GEN.format(max_tokens=max_tokens)
    s += "        backend_args:\n"
    if few_shot:
        s += "          few_shot_num: 0\n"
    s += "          judge_strategy: rule\n          dataset_hub: modelscope\n"
    return s

# ---- 1. Tekstovyy nabor: kazhdyy dataset otdelnoy syuitoy, chtoby zapuskat po ocheredi
text = TARGET.format(conc=4) + "\nsuites:\n"
text += "  aime25:\n    jobs:\n" + job("aime25", "aime25", 4, AIME_MAX, few_shot=False)
text += "  aime26:\n    jobs:\n" + job("aime26", "aime26", 4, AIME_MAX, few_shot=False)
text += "  ifbench:\n    jobs:\n" + job("ifbench", "ifbench", 4, IFB_MAX)
text += "  gpqa:\n    jobs:\n" + job("gpqa_diamond", "gpqa_diamond", 4, GPQA_MAX)
text += "  aime_both:\n    jobs:\n" + job("aime25", "aime25", 4, AIME_MAX, few_shot=False) \
      + job("aime26", "aime26", 4, AIME_MAX, few_shot=False)
text += RUNTIME
(OUT / "text35.yaml").write_text(text, encoding="utf-8")

# ---- 2. NIAH: puti perepisany, vyborka avtorskaya (temp 0, bez rassuzhdeniy)
def niah(jid, cmin, cmax, intervals, depths, subsets="[english, chinese]"):
    return f"""      - id: {jid}
        backend: evalscope
        dataset: needle_haystack
        target: qwen35_api
        max_concurrency: 1
        generation:
          temperature: 0
          max_tokens: 512
          seed: 42
          extra_body:
            enable_thinking: false
        backend_args:
          subset_list: {subsets}
          judge_strategy: rule
          dataset_args:
            local_path: {CORPUS}
            extra_params:
              context_lengths_min: {cmin}
              context_lengths_max: {cmax}
              context_lengths_num_intervals: {intervals}
              document_depth_percent_min: 0
              document_depth_percent_max: 100
              document_depth_percent_intervals: {depths}
              tokenizer_path: {TOKENIZER}
              show_score: true
"""

nia = TARGET.format(conc=1) + "\nsuites:\n"
nia += "  standard:\n    jobs:\n" + niah("needle_standard", 1000, 32000, 10, 10)
nia += "  long_64k:\n    jobs:\n"  + niah("needle_64k",  65536, 65536, 1, 11)
nia += "  long_128k:\n    jobs:\n" + niah("needle_128k", 131072, 131072, 1, 11)
nia += "  long_260k:\n    jobs:\n" + niah("needle_260k", 260000, 260000, 1, 11)
nia += "  smoke:\n    jobs:\n"     + niah("needle_smoke", 8192, 8192, 1, 1, "[english]")
nia += RUNTIME
(OUT / "niah35.yaml").write_text(nia, encoding="utf-8")

print(f"napisano: ctx={CTX} aime={AIME_MAX} ifbench={IFB_MAX} gpqa={GPQA_MAX}")
