# DeepSeek-V4.1-Flash 2× DGX Spark

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

Apply on `deepseek-v41-zig` @ `4e80f05`, without 0005 (see its row):

```
git am patches/000[1-4]*.patch patches/000[6-9]*.patch
```

With this set, replies are identical to the stock build (15/15 in the agent replay below).

| Patch | Change | Default |
|---|---|---|
| 0001 | The prompt cache keys a request with images by the images' bytes. The same image resumes, a different image never gets a false hit, and image spans before the resume point skip the vision tower. | on |
| 0002 | When the shared KV pool needs a kept state's rows, each rank writes its half to NVMe (O_DIRECT) instead of dropping it, and reads it back on resume. | `TF_DS_SPILL_DIR=<dir>` on both ranks (cap `TF_DS_SPILL_GIB`, default 48) |
| 0003 | `tf-dsv41-lanes`: per-request `fix_k` and per-request round costs (a measurement tool). | tool |
| 0004 | Adaptive top-k for the routed experts of decode windows. Experimental: quality not validated yet. | off (`TF_DS_TOPK_TAU`) |
| 0005 | Prompt chunks of 4,096 rows instead of 2,048: cold prompts 5–7 % faster, ~2.5 GB more arena per rank. Not recommended: the prefill sums run in a different order, so replies differ from stock after a few words (0/15 identical; quality not measured), and kept states resume on a 4,096-token grid, so a cached agent turn redoes ~1,000 more tokens (+24 % time to first token). | don't apply |
| 0006 | The RDMA ring's queue pairs take the port's active MTU (at most 4,096). The ring then works at Ethernet MTU 1500. | on with `--rdma` |
| 0007 | The RDMA ring picks the port's RoCE v2 IPv4 GID. Stock code uses index 5, which is empty on a usual Spark link (ours: 3), so the ring failed with `VerbsFailed`. | on with `--rdma` (`TF_DS_RDMA_GID` overrides) |
| 0008 | The tokenizer memoizes each text segment between added tokens (a chat turn tokenizes the whole conversation twice). A cached agent turn of 75K–100K tokens prepares in 0.03 s instead of 0.14 s. Same ids. | on |
| 0009 | The prompt indexer reads each row block's `vis` from a 16-byte-aligned address. At ~80K tokens of context a block is an odd number of rows (409–419), and the kit's 16-byte loads then hit an 8-byte-aligned view. | on |

## Results

**Decode** (`tensorfold-native serve`, same settings as production, greedy, 384 tokens, median of 3; MTU 9000). The
decode all-gathers go over the engine's RDMA ring instead of NCCL (patches 0006–0007 make the ring connect on a stock
Spark link).

| Prompt | NCCL | RDMA ring | Gain | Same text |
|---|---:|---:|---:|:---:|
| code, one stream | 90.4 tok/s | **99.1 tok/s** | +10 % | yes |
| prose, one stream | 55.6 tok/s | **61.4 tok/s** | +10 % | yes |
| structured, one stream | 127.6 tok/s | **138.2 tok/s** | +8 % | yes |
| 4 streams, aggregate | 107.3 tok/s | **114.6 tok/s** | +7 % | yes |

One gather of 1 row (20 KB) takes 9 µs on the ring against 21 µs with NCCL; a decode round does 80 of them. Engine
harness rows (per-round costs, token hashes): `results/decode_nccl_vs_rdma.jsonl`.

Stock, the ring's queue pairs use a fixed 4,096-byte path MTU. At Ethernet MTU 1500 the RoCE MTU is 1,024 and
every gather fails, which is why the stock setup runs NCCL. With patch 0006 the ring runs at MTU 1500 within 1 % of
MTU 9000, same tokens (`results/decode_mtu1500_vs_9000.jsonl`). No MTU change needed. Patch 0007 then finds the right
GID by itself (`show_gids`: the RoCE v2 row with your IPv4 address). On both ranks:

```
# rank 0 (tensorfold-native serve)      # rank 1 (tf-dsv41-lanes)
TF_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1   --rdma rocep1s0f1,roceP2p1s0f1
```

**Context length** (harness, the recommended set, one stream, greedy, 256 tokens after a long document;
`results/ctx_sweep.jsonl`):

| Prompt | Cold prefill | Prefill rate | Decode | Tokens a round | Forward a round |
|---|---:|---:|---:|---:|---:|
| 4K | 2.0 s | 2,040 tok/s | 97 tok/s | 4.20 | 39.0 ms (5.0 rows) |
| 32K | 13.4 s | 2,440 tok/s | 71 tok/s | 2.86 | 36.1 ms (3.9 rows) |
| 128K | 57.1 s | 2,300 tok/s | 61 tok/s | 2.42 | 35.2 ms (3.4 rows) |
| 512K | 270 s | 1,940 tok/s | 68 tok/s | 3.01 | 38.8 ms (4.0 rows) |

Decode speed follows the text (tokens a round) more than the context: a 4-row round costs 35 ms at 4K, 37.5 ms at
128K and 39 ms at 512K (sparse attention). Prefill goes from 0.41 ms a token below 32K to 0.54 ms between 128K and
512K. At 128K with the served sampling (T=1, top-p 0.95, top-k 20): 65 tok/s.

**Prefill** (`tf-dsv41-lanes` harness, MTU 9000, one request at a time). Patch 0005, not recommended: the first token
is the same, but replies then differ (the prefill sums run in another order):

| Prompt | 2,048-row chunks | 4,096-row chunks | Gain |
|---|---:|---:|---:|
| 8K tokens | 3.75 s | **3.53 s** | +6 % |
| 32K tokens | 13.32 s | **12.64 s** | +5 % |
| 128K tokens | 56.90 s | **53.95 s** | +5 % |

Raw rows: `results/prefill_chunk2048_vs_4096.jsonl`.

**Agent replay** (`bench/agent_replay.py` run on the head node, `--parallel 4`, greedy, thinking off; only the engine
binary changes). Three sessions of 72K / 80K / 89K tokens (one carries an image) take 5 turns each in rotation.
Every turn appends the previous reply plus ~2.7K tokens of tool output. The KV pool is 262,144 positions, so the
sessions crowd each other out, as several agents do on the 1M pool.

| | Stock | Patched (no 0005) | Patched + 0005 |
|---|---:|---:|---:|
| Time to first token, turns 2–5, mean | 31.6 s | **2.1 s** (15×) | 2.6 s |
| Time to first token, turns 2–5, max | 45.6 s | **2.6 s** (17×) | 3.4 s |
| Prompt tokens served from cache, turns 2–5 | 21 % | **96 %** | 95 % |
| Wall time, 15 turns | 504 s | **148 s** (3.4×) | 146 s |
| Replies identical to the stock build | | **15 / 15** | 0 / 15 |

Raw rows: `results/agent_replay_local_{stock,patched,patched_0005}.json`. Run from a remote client over a high-latency
link, each turn took 2–4 s longer: the 300–430 KB request went out on a fresh TCP connection every turn. Keep the
agents near the server, or reuse connections.

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
- **Decode round** (CUDA graphs on, one stream), forward ms by rows verified:

  | Rows | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---:|---:|---:|---:|---:|---:|
  | NCCL gathers | 24.6 | 29.7 | 33.9 | 38.2 | 42.0 | 45.7 |
  | RDMA ring | 22.1 | 27.2 | 31.3 | 35.2 | 39.2 | 42.3 |
  | no gathers at all (wrong outputs, timing only) | 22.0 | 26.7 | 30.5 | 34.0 | 37.6 | 40.9 |

  Plus 3.6 ms for the drafter pass. A 1-row round reads ~4.3 GB of weights per rank (attention 1.7, routed experts
  1.5, shared expert 0.4, head 0.25), i.e. ~18 ms at DRAM speed: with the ring, communication is no longer the limit.
- **Inside a 1-row round** (GPU clocks at every phase, inside the CUDA graphs; `results/decode_round_phases.json`):

  | Phase | ms | Weights read | At 238 GB/s |
  |---|---:|---:|---:|
  | Routed and shared experts | 8.4 | 1.94 GB | 8.1 ms |
  | Attention projections (wq_a, wkv, wq_b, wo_a, wo_b) | 7.1 | 1.74 GB | 7.3 ms |
  | Router, mHC mixes, Engram | 1.7 | 0.40 GB | 1.7 ms |
  | TP2 gathers (76 × 23 µs) | 1.8 | | |
  | Sparse attention core, indexer, compressor | 1.1 | | |

  The weight phases run at the DRAM limit (the L2 prefetch even beats it on wo_a and wo_b). The gathers are mostly
  hidden: the prefetch fills L2 meanwhile, and removing them saves 0.1 ms. The round runs at ~90 % of DRAM bandwidth,
  so decode speed now comes from fewer bytes per generated token, not from kernel tuning.
- **Prefill chunk** (2,048 rows): MoE 43 % (gate/up 22 %, down 14 %, routing and input rotation 7 %), attention
  28 %, NCCL gathers 13 %, mHC mixes 10 %, Engram 5 %. Nsight Compute on the MoE kernels: ~31 instructions issued
  per tensor-core MMA (EXL3 trellis decoding), 33 % / 17 % occupancy, latency-bound, not DRAM- or tensor-bound.

## Tried, no gain

- NCCL protocols (LL, LL128, auto) and tuning (channels, QPs per connection, GPUDirect level, buffer size).
- MTU 9000: decode unchanged with NCCL, and with patch 0006 the RDMA ring does not need it either.
- The RDMA ring for the large prefill gathers: 2.97 ms for 42 MB against 2.25 ms with NCCL.
- Prompt MoE kernels over 128 rows a program (bit-exact, written for this): 15-30 % slower, since most experts get
  64 rows or fewer per chunk.
- Prefill all-gathers in row pieces on a side stream, under the mHC mixes: within noise (±1.5 %).
- Coupled drafts at T>0 (the drafter draws with the target's own keyed Gumbel noise, sharpened by a factor λ;
  outputs unchanged): −6 % rounds on code with thinking, nothing on prose, −1 to −2 % overall. The drafter finds the
  target's top token, not the shape of its distribution.
- 4,096-row prompt chunks (patch 0005): faster cold prefill, but different replies and slower cached agent turns.

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
- several streams per round (4 and 8 streams), longer copy windows for code and agent output;
- a round-cost table measured with the ring (the served one is 8-10 ms too high);
- a bigger cache of the drafter's Markov bias rows (a miss reads 66 MB);
- adaptive top-k, only if quality and security-knowledge scores hold.

A decode round is already at ~90 % of DRAM bandwidth and each extra verified row costs ~4 ms, mostly new experts:
what is left is in tokens per weight read (drafts, streams), not in kernels.

## Credits

DeepSeek (model, MIT), Mia-AiLab (EXL3 quant, MIT), turboderp (EXL3), ashhart (TensorFold, Apache-2.0), bertholomus
(the `dsv41` TP2 family, Apache-2.0). Patches and scripts here: Apache-2.0 (`LICENSE.md`, `NOTICE`).
