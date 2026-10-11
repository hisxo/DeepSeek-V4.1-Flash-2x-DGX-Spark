#!/usr/bin/env python3
"""Image prompt-cache gate: for an image early and an image late in a long prompt, a resend must (1) resume from the
cache and (2) give the same greedy reply as the fresh pass; a different image must give its own reply."""
import json, sys, time, urllib.request, base64, zlib, struct
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
def png(seed, w=48, h=48):
    raw = b"".join(b"\x00" + bytes([v for x in range(w) for v in (((x * seed) % 256), ((y * 7 + seed * 31) % 256), (seed * 50) % 256)]) for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
filler = " ".join(f"item{i} is the {i%7}th colour of block {i//7};" for i in range(1500))
def url(seed): return "data:image/png;base64," + base64.b64encode(png(seed)).decode()
def ask(order, seed, tag):
    img = {"type": "image_url", "image_url": {"url": url(seed)}}; txt = {"type": "text", "text": "Reference notes:\n" + filler}
    q = {"type": "text", "text": "Describe the image's dominant colours in one sentence, then say how many items the notes list."}
    content = [img, txt, q] if order == "early" else [txt, img, q]
    body = {"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": content}], "max_tokens": 48,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    u = d.get("usage", {}); c = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    text = d["choices"][0]["message"].get("content")
    print(f"{tag:14s} {time.time()-t:6.2f}s prompt {u.get('prompt_tokens')} cached {c} :: {text!r}", flush=True)
    return text, c
ok = True
for order in ("early", "late"):
    a, _ = ask(order, 3, order + "/img3#1"); b, cb = ask(order, 3, order + "/img3#2"); c, _ = ask(order, 9, order + "/img9#1")
    same = a == b; hit = (cb or 0) > 0; diff = c != a
    print(f"  {order}: resend equal={same} resumed={hit} other-image differs={diff}")
    ok = ok and same and hit and diff
print("GATE", "PASS" if ok else "FAIL")
