#!/usr/bin/env python3
"""Do live streams keep decoding while a long prompt prefills? (TF_DS_PF_YIELD, lane_host yield between chunks)

  python3 interleave_test.py BASE OUT.json [--long 32768] [--tokens 1536] [--delay 3]

Three greedy streams (the set-b prompts, ignore_eos, --tokens each) start together; --delay seconds later a fresh
--long-token prompt (no cache hit) is sent with max_tokens 32. Recorded: each stream's tokens landed in the window
[long request sent, long request's first token] (events in the window x the stream's tokens an event: a round's
tokens come in one event), its total tokens and seconds, the long request's time to first token, the wall time of all
four, and the sha256 of every text (identical across TF_DS_PF_YIELD settings: the rounds
and chunks are the same computations in another order).
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kit_bench as kb  # noqa: E402


def stream_times(c: kb.Client, body: dict, path: str) -> dict:
    body = dict(body, model=c.model, stream=True, stream_options={"include_usage": True})
    start = time.perf_counter()
    marks, pieces, usage = [], [], None
    with c._req(path, body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if piece:
                    marks.append(time.perf_counter())
                    pieces.append(piece)
    text = "".join(pieces)
    return {"start": start, "end": time.perf_counter(), "marks": marks, "text": text,
            "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "tokens": int(usage["completion_tokens"]) if usage else len(marks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--long", type=int, default=32768)
    ap.add_argument("--tokens", type=int, default=1536)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20261123)
    a = ap.parse_args()
    c = kb.Client(a.base, "DeepSeek-V4.1-Flash", None)
    prompts = [c.chat_ids(p) for _, p in kb.DECODE_PROMPTS_B]
    long_ids = kb.filler_ids(c, a.long, a.seed)
    results = [None] * 3
    long_res = {}

    def worker(i):
        results[i] = stream_times(c, {"prompt": prompts[i], "max_tokens": a.tokens, "temperature": 0, "ignore_eos": True}, "/v1/completions")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    time.sleep(a.delay)
    long_res.update(stream_times(c, {"prompt": long_ids, "max_tokens": 32, "temperature": 0}, "/v1/completions"))
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    l0 = long_res["start"]
    l1 = long_res["marks"][0] if long_res["marks"] else long_res["end"]
    rows = []
    for name, r in zip([n for n, _ in kb.DECODE_PROMPTS_B], results):
        # the server sends a round's tokens in one event: tokens in the window = events in the window x tokens an event
        events = len(r["marks"])
        during = sum(1 for m in r["marks"] if l0 <= m <= l1)
        per_event = r["tokens"] / events if events else 0
        tokens_during = round(during * per_event)
        rows.append({"stream": name, "tokens": r["tokens"], "events": events, "seconds": round(r["end"] - r["start"], 2),
                     "tok_s": round((r["tokens"] - 1) / (r["marks"][-1] - r["marks"][0]), 1) if len(r["marks"]) > 1 else None,
                     "events_during_prefill": during, "tokens_during_prefill": tokens_during,
                     "tok_s_during_prefill": round(tokens_during / (l1 - l0), 1) if l1 > l0 else None, "sha": r["sha"]})
    out = {"long_tokens": len(long_ids), "long_ttft_s": round(l1 - l0, 2), "long_sha": long_res["sha"], "wall_s": round(wall, 2),
           "streams": rows, "env": {k: v for k, v in os.environ.items() if k.startswith("TF_")}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "env"}))


if __name__ == "__main__":
    main()
