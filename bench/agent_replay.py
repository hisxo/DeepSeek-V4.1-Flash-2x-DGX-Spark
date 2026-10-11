#!/usr/bin/env python3
"""Agentic replay: several long coding-agent sessions take turns on one server, the way a few concurrent agents do.
Each session starts from a large context (one carries an image), and every turn appends the previous reply plus a
tool result of a few thousand tokens. Greedy, thinking off, streamed: time to first token (TTFT) per turn, cached
prompt tokens, and each reply's text (to check that a faster resume gives the same reply as a full prefill).

  agent_replay.py BASE OUT.json [--sessions 3] [--turns 4] [--base-words 33000] [--tool-words 1200]
"""
import argparse, base64, json, random, struct, sys, time, urllib.request, zlib

VOC = ("def class return import self value index buffer request token stream cache layer kernel thread "
       "async await error result config model tensor offset length chunk shape batch queue worker").split()

def png(seed, w=64, h=64):
    raw = b"".join(b"\x00" + bytes([v for x in range(w) for v in ((x * seed) % 256, (y * 5 + seed * 17) % 256, (x ^ y) % 256)]) for y in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b"")

def text(rng, words):
    lines, line = [], []
    for _ in range(words):
        line.append(rng.choice(VOC) + str(rng.randrange(1000)))
        if len(line) == 12:
            lines.append("    " + " ".join(line)); line = []
    return "\n".join(lines)

def stream(base, body):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; out = []; usage = {}; extra = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for k, v in d.items():
                if k not in ("choices", "usage", "id", "object", "created", "model") and v: extra[k] = v
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    if ttft is None: ttft = time.time() - t0
                    out.append(piece)
    return {"ttft_s": ttft, "total_s": time.time() - t0, "text": "".join(out), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, "server": extra}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("base"); p.add_argument("out")
    p.add_argument("--sessions", type=int, default=3); p.add_argument("--turns", type=int, default=4)
    p.add_argument("--base-words", type=int, default=33000); p.add_argument("--tool-words", type=int, default=1200)
    p.add_argument("--max-tokens", type=int, default=96)
    a = p.parse_args()
    sessions = []
    for s in range(a.sessions):
        rng = random.Random(100 + s)
        first = [{"type": "text", "text": f"Project {s} source tree:\n" + text(rng, a.base_words + 4000 * s)}]
        if s == a.sessions - 1:   # one session carries a screenshot from its first turn on
            first.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png(7 + s)).decode()}})
        first.append({"type": "text", "text": "Task: find where the cache is invalidated and summarise it in two sentences."})
        sessions.append({"rng": rng, "messages": [{"role": "system", "content": "You are a coding agent. Be brief."},
                                                  {"role": "user", "content": first}]})
    rows = []
    for turn in range(a.turns + 1):
        for s, ss in enumerate(sessions):
            body = {"model": "DeepSeek-V4.1-Flash", "messages": ss["messages"], "max_tokens": a.max_tokens, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}}
            try:
                r = stream(a.base, body)
            except Exception as e:  # an HTTP error or a dropped stream: kept as a failed turn
                r = {"ttft_s": None, "total_s": 0.0, "text": "", "prompt_tokens": None, "cached_tokens": 0,
                     "server": {"exception": repr(e), "body": (e.read().decode(errors="replace")[:500] if hasattr(e, "read") else "")}}
            r.update(session=s, turn=turn)
            rows.append(r)
            rt = r["server"].get("tensorfold") or r["server"].get("runtime") or {}
            if r["prompt_tokens"] is None or r["ttft_s"] is None:
                print(f"s{s} t{turn} FAILED: {json.dumps(r['server'])[:600]} text {r['text'][:200]!r}", flush=True)
                r["ttft_s"] = r["ttft_s"] or r["total_s"]; r["prompt_tokens"] = r["prompt_tokens"] or 0
            else:
                print(f"s{s} t{turn} prompt {r['prompt_tokens']:>7} cached {r['cached_tokens']:>7} ttft {r['ttft_s']:7.2f}s total {r['total_s']:7.2f}s"
                      f" server: prefill {rt.get('prefill_seconds')} ttft {rt.get('time_to_first_token')}", flush=True)
            ss["messages"] = ss["messages"] + [{"role": "assistant", "content": r["text"]},
                                               {"role": "user", "content": "Tool output (grep -n):\n" + text(ss["rng"], a.tool_words) + "\nContinue."}]
    later = [r for r in rows if r["turn"] > 0]
    summary = {"turns": len(rows), "later_turns": len(later),
               "later_ttft_s_mean": sum(r["ttft_s"] for r in later) / len(later),
               "later_ttft_s_max": max(r["ttft_s"] for r in later),
               "later_cached_ratio": sum(r["cached_tokens"] for r in later) / sum(r["prompt_tokens"] for r in later),
               "wall_s": sum(r["total_s"] for r in rows)}
    print(json.dumps(summary))
    args = {k: v for k, v in vars(a).items() if k not in ("base", "out")}
    json.dump({"args": args, "summary": summary, "rows": rows}, open(a.out, "w"), indent=1)

main()
