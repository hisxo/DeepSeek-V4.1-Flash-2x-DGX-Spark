#!/usr/bin/env python3
"""Spill gate: on a small pool, three long sessions crowd each other out; resending the first two must resume (from
disk when spilled) and give the fresh pass's greedy reply. Usage: spill_check.py BASE [WORDS]"""
import json, sys, time, random, urllib.request
BASE = sys.argv[1]; WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 36000
VOC = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu river stone cloud ember maple quartz violet amber cobalt".split()
def prompt(seed):
    r = random.Random(seed)
    return f"Session {seed}. Notes:\n" + " ".join(r.choice(VOC) + str(r.randrange(100)) for _ in range(WORDS)) + "\nQuestion: which three words appear first in the notes, and what is the session number?"
def ask(seed, tag):
    body = {"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": prompt(seed)}], "max_tokens": 40,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
    u = d.get("usage", {}); c = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    text = d["choices"][0]["message"].get("content")
    print(f"{tag:6s} {time.time()-t:7.2f}s prompt {u.get('prompt_tokens')} cached {c} :: {text!r}", flush=True)
    return text, c
a1, _ = ask(1, "A#1"); b1, _ = ask(2, "B#1"); c1, _ = ask(3, "C#1")
a2, ca = ask(1, "A#2"); b2, cb = ask(2, "B#2")
ok = a1 == a2 and b1 == b2 and ca > 0 and cb > 0
print("GATE", "PASS" if ok else "FAIL", {"A_equal": a1 == a2, "B_equal": b1 == b2, "A_cached": ca, "B_cached": cb})
