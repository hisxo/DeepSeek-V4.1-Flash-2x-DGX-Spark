# DeepSeek-V4.1-Flash on 2× DGX Spark: engine patches

Patches for the TP2 Zig engine that serves DeepSeek-V4.1-Flash (EXL3 2.9 bpw) across two NVIDIA GB10 nodes:
the `dsv41` family of [bertholomus/TensorFold, branch `deepseek-v41-zig`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig),
weights [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw).

## Why

24 h of real coding-agent traffic on the stock build (1,853 requests, median prompt 163K tokens, 80 % of prompt
tokens already cached):

| Prompt-cache outcome | Requests | Prompt tokens prefilled | Share of prefill |
|---|---:|---:|---:|
| resumed from a kept state | 1,608 | 15.4 M | 24 % |
| **request carried an image: cache never consulted** | 79 | 25.9 M | **40 %** |
| **kept state overwritten in the shared KV pool: resume failed** | 106 | 22.0 M | **34 %** |
| real miss | 65 | 1.5 M | 2 % |

Three quarters of all prefill time redid work that had already been done. Those turns waited 150–240 s for their
first token.

## Patches

Apply on `deepseek-v41-zig` @ `4e80f05` with `git am patches/*.patch`.

| Patch | Change | Default |
|---|---|---|
| 0001 | The prompt cache keys a request with images by the images' bytes. The same image resumes, a different image never gets a false hit, and image spans before the resume point skip the vision tower. | on |
| 0002 | When the shared KV pool needs a kept state's rows, each rank writes its half to NVMe (O_DIRECT) instead of dropping it, and reads it back on resume. | `TF_DS_SPILL_DIR=<dir>` on both ranks (cap `TF_DS_SPILL_GIB`, default 48) |
| 0003 | `tf-dsv41-lanes`: per-request `fix_k` and per-request round costs (a measurement tool). | tool |
| 0004 | Adaptive top-k for the routed experts of decode windows. Experimental: quality not validated yet. | off (`TF_DS_TOPK_TAU`) |

## Results

**Decode** (`tensorfold-native serve`, same settings as production, greedy, 384 tokens, median of 3; MTU 9000). The
decode all-gathers go over the engine's RDMA ring instead of NCCL. No code change, only settings.

| Prompt | NCCL | RDMA ring | Gain | Same text |
|---|---:|---:|---:|:---:|
| code, one stream | 90.4 tok/s | **99.1 tok/s** | +10 % | yes |
| prose, one stream | 55.6 tok/s | **61.4 tok/s** | +10 % | yes |
| structured, one stream | 127.6 tok/s | **138.2 tok/s** | +8 % | yes |
| 4 streams, aggregate | 107.3 tok/s | **114.6 tok/s** | +7 % | yes |

One gather of 1 row (20 KB) takes 9 µs on the ring against 21 µs with NCCL; a decode round does 80 of them. Engine
harness rows (per-round costs, token hashes): `results/decode_nccl_vs_rdma.jsonl`.

The ring's queue pairs use a fixed 4,096-byte path MTU. At Ethernet MTU 1500 the RoCE MTU is 1,024 and every
gather fails, which is why the stock setup runs NCCL. Set MTU 9000 on both RoCE ports of both nodes, then:

```
# rank 0 (tensorfold-native serve)      # rank 1 (tf-dsv41-lanes)
TF_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1   --rdma rocep1s0f1,roceP2p1s0f1
```

**Agent replay** (`bench/agent_replay.py`, patches 0001–0002, `--parallel 4`, greedy, thinking off; only the engine
binary changes). Three sessions of 72K / 80K / 89K tokens (one carries an image) take 5 turns each in rotation.
Every turn appends the previous reply plus ~2.7K tokens of tool output. The KV pool is 262,144 positions, so the
sessions crowd each other out, as several agents do on the 1M pool.

| | Stock | Patched |
|---|---:|---:|
| Time to first token, turns 2–5, mean | 33.8 s | **4.9 s** (6.9×) |
| Time to first token, turns 2–5, max | 49.4 s | **5.5 s** (9.1×) |
| Prompt tokens served from cache, turns 2–5 | 21 % | **96 %** |
| Wall time, 15 turns | 536 s | **192 s** (2.8×) |
| Replies identical to the stock build | | **15 / 15** |

Raw rows: `results/agent_replay_{stock,patched}.json`.

**Gates**

- `bench/img_cache_check.py`: a 20K-token prompt with an image, image first or last. Patched resend: 11.4 s → 2.8 s
  with the same greedy reply. A different image never reuses the image span, and in that case only the shared text
  prefix is resumed.
- `bench/spill_check.py` on a 131,072-position pool: three ~90K-token sessions evict each other. Patched resend: 2.9 s
  (first send 43 s) with the same reply. A 90K-token state is 76 MiB on disk, written in 22–62 ms and read back in
  11–22 ms. The stock build drops the evicted state, so its resend prefills from scratch.

## Where the time goes

These measurements guide the next patches.

- **GB10** (`bench/hwbench.py`, `results/hw_gb10.json`): 238 GB/s DRAM read, 95 TFLOPS BF16, 199 FP8, 339 NVFP4.
- **Decode round** (CUDA graphs on, one stream):

  | Rows verified | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---:|---:|---:|---:|---:|---:|
  | Forward, ms | 24.7 | 29.6 | 34.0 | 38.5 | 42.2 | 45.6 |

  Add 4.1 ms for the drafter pass. About 16 ms of the 1-row round is weight streaming at DRAM speed and ~5 ms is
  NCCL all-gathers (80 a round); the RDMA ring saves ~3 ms of that. The decode is bandwidth-bound.
- **Prefill chunk** (2,048 rows, 32K prompt): MoE 46 %, attention 25 %, NCCL gathers 14 %, mHC mixes 15 %. That is
  ~2.5× above what the bandwidth and the tensor cores allow.

## Build and run

Build and run as described in the upstream recipe
([bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)),
after `git am`. Inside `nvcr.io/nvidia/pytorch:26.07-py3`:

```
zig build install native -Dtarget=aarch64-linux-gnu -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix <OUT>
```

Add `TF_DS_SPILL_DIR=<dir on local NVMe>` to both ranks' environment, and mount that directory into the containers.

```
python3 bench/agent_replay.py http://<head>:<port> out.json
python3 bench/img_cache_check.py http://<head>:<port>
python3 bench/spill_check.py http://<head>:<port>      # on a small pool, e.g. --context 131072
```

## Next

Work in progress, in this repository:
- prefill: 4,096-row chunks; overlap the all-gathers with compute (NCCL tops out at ~19 GB/s on this link);
- several streams per round, and longer copy windows for code and agent output;
- tokenizer memoization for long agent sessions;
- adaptive top-k, only if quality and security-knowledge scores hold.

## Credits

DeepSeek (model, MIT), Mia-AiLab (EXL3 quant, MIT), turboderp (EXL3), ashhart (TensorFold, Apache-2.0), bertholomus
(the `dsv41` TP2 family, Apache-2.0). Patches and scripts here: Apache-2.0 (`LICENSE.md`, `NOTICE`).
