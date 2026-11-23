#!/usr/bin/env python3
"""Where does a cached turn's time to first token go? (client TTFT = network + host work + prefill + first draw)

  python3 ttft_decomp.py BASE OUT.json [--base-tokens 60000] [--turns 6] [--turn-tokens 1500]

One conversation: a system note of ~base-tokens filler, then `turns` user turns of ~turn-tokens new text each, greedy,
8 tokens a reply, thinking off, streamed. Around every request the server's /metrics counters are read:
  prefill_s  = d request_prefill_seconds_sum   (the prompt pass alone, server clock)
  server_ttft = d time_to_first_token_seconds_sum (arrival to first token, server clock; no queue here)
so host work = server_ttft - prefill_s (parse, tokenize, cache lookup, lane admission, first draw) and
network + client = client_ttft - server_ttft. Also the prompt tokens served from cache (d prompt_tokens - prefilled).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kit_bench as kb  # noqa: E402

KEYS = ("request_prefill_seconds_sum", "time_to_first_token_seconds_sum", "prompt_tokens_total", "request_latency_seconds_count")


def metrics(base):
    txt = urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10).read().decode()
    out = {}
    for k in KEYS:
        m = re.search(r"^tensorfold:" + k + r" ([0-9.eE+-]+)", txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--base-tokens", type=int, default=60000)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--turn-tokens", type=int, default=1500)
    a = ap.parse_args()
    c = kb.Client(a.base, "DeepSeek-V4.1-Flash", None)
    import random
    rng = random.Random(7)
    def words(n):
        return " ".join(rng.choice(kb.WORDS) for _ in range(int(n * 0.9)))
    system = "Background notes, context only.\n" + words(a.base_tokens)
    messages = [{"role": "system", "content": system}]
    rows = []
    for t in range(a.turns):
        messages.append({"role": "user", "content": f"Turn {t}. Reply with one word.\n" + words(a.turn_tokens)})
        m0 = metrics(a.base)
        r = c.stream({"messages": messages, "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, "/v1/chat/completions")
        m1 = metrics(a.base)
        row = {"turn": t, "prompt_tokens": r["prompt_tokens"], "client_ttft_s": round(r["ttft_s"], 3),
               "server_ttft_s": round(m1["time_to_first_token_seconds_sum"] - m0["time_to_first_token_seconds_sum"], 3),
               "prefill_s": round(m1["request_prefill_seconds_sum"] - m0["request_prefill_seconds_sum"], 3)}
        row["host_s"] = round(row["server_ttft_s"] - row["prefill_s"], 3)
        row["net_client_s"] = round(row["client_ttft_s"] - row["server_ttft_s"], 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
        messages.append({"role": "assistant", "content": r["text"]})
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
