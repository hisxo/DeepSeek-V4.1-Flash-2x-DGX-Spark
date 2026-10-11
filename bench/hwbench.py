#!/usr/bin/env python3
"""GB10 hardware micro-benchmarks: memory bandwidth (read / copy / write), tensor-core TFLOPS by dtype, L2, clocks."""
import torch, time, json, sys
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
p = torch.cuda.get_device_properties(0)
out = {"name": p.name, "sms": p.multi_processor_count, "cc": f"{p.major}.{p.minor}", "l2_mb": getattr(p, "L2_cache_size", 0) / 2**20,
       "total_gib": p.total_memory / 2**30}
def timeit(fn, reps=20, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps / 1e3
# --- bandwidth
bw = {}
for gib in (0.25, 1, 4):
    n = int(gib * 2**30) // 4
    a = torch.empty(n, dtype=torch.float32, device=dev).uniform_(); b = torch.empty_like(a)
    t = timeit(lambda: a.sum()); bw[f"read_sum_{gib}GiB"] = n * 4 / t / 1e9
    t = timeit(lambda: b.copy_(a)); bw[f"copy_{gib}GiB(r+w)"] = 2 * n * 4 / t / 1e9
    t = timeit(lambda: b.fill_(1.0)); bw[f"write_{gib}GiB"] = n * 4 / t / 1e9
    ai = a.view(torch.int32)
    t = timeit(lambda: torch.bitwise_xor(ai[: n // 2], ai[n // 2 :], out=b.view(torch.int32)[: n // 2])); bw[f"xor_{gib}GiB(2r+1w)"] = 1.5 * n * 4 / t / 1e9
    del a, b, ai; torch.cuda.empty_cache()
# L2-resident read
a = torch.empty(4 * 2**20 // 4, dtype=torch.float32, device=dev).uniform_()
t = timeit(lambda: a.sum(), reps=200); bw["read_sum_4MiB(L2)"] = a.numel() * 4 / t / 1e9
out["bandwidth_GBps"] = bw
# --- tensor cores
tf = {}
for M, N, K in ((8192, 8192, 8192), (4096, 4096, 4096), (2048, 5120, 1152), (32, 5120, 1152), (6, 5120, 1152)):
    for dt in (torch.bfloat16, torch.float16):
        x = torch.randn(M, K, device=dev, dtype=dt); w = torch.randn(K, N, device=dev, dtype=dt)
        t = timeit(lambda: x @ w); tf[f"{str(dt)[6:]}_{M}x{N}x{K}"] = 2 * M * N * K / t / 1e12
    try:
        x8 = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn); w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn).t()
        one = torch.ones((), device=dev)
        t = timeit(lambda: torch._scaled_mm(x8, w8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)); tf[f"fp8_{M}x{N}x{K}"] = 2 * M * N * K / t / 1e12
    except Exception as ex: tf[f"fp8_{M}x{N}x{K}"] = f"ERR {type(ex).__name__}: {str(ex)[:80]}"
try:
    M = N = K = 8192
    x4 = torch.randint(0, 255, (M, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    w4 = torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    sa = torch.ones(M * K // 16, device=dev).to(torch.float8_e4m3fn); sb = torch.ones(N * K // 16, device=dev).to(torch.float8_e4m3fn)
    t = timeit(lambda: torch._scaled_mm(x4, w4.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)); tf["nvfp4_8192^3"] = 2 * M * N * K / t / 1e12
except Exception as ex: tf["nvfp4_8192^3"] = f"ERR {type(ex).__name__}: {str(ex)[:120]}"
out["tflops"] = tf
print(json.dumps(out, indent=1))
