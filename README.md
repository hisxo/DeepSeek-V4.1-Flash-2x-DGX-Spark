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

Apply on `deepseek-v41-zig` @ `4e80f05`:

```
git am patches/*.patch
```

Without 0005 and with the defaults, replies are identical to the stock build (15/15 in the agent replay below). The
recommended settings add 0005, `TF_DS_TOPK_TAU=0.45` and `TF_DS_EXACT_TAIL=1`, which change rounding or skip weak
experts: replies differ, quality and security-knowledge scores do not (Results).

| Patch | Change | Default |
|---|---|---|
| 0001 | The prompt cache keys a request with images by the images' bytes. The same image resumes, a different image never gets a false hit, and image spans before the resume point skip the vision tower. | on |
| 0002 | When the shared KV pool needs a kept state's rows, each rank writes its half to NVMe (O_DIRECT) instead of dropping it, and reads it back on resume. | `TF_DS_SPILL_DIR=<dir>` on both ranks (cap `TF_DS_SPILL_GIB`, default 48) |
| 0003 | `tf-dsv41-lanes`: per-request `fix_k` and per-request round costs (a measurement tool). | tool |
| 0004 | Adaptive top-k for the routed experts of decode windows: a routed expert whose weight is under τ × the row's top routed weight is dropped (at least 4 of the 6 kept, the rest renormalized). At τ 0.45 greedy decode is 0–15 % faster (code +15 %, prose +9 %, structured ≈) and thinking at T=1 +11 %; replies change slightly (~3 % of reasoned answers) and quality and security-knowledge scores are unchanged on 8 evals (below). | `TF_DS_TOPK_TAU=0.45` (recommended) |
| 0005 | Prompt chunks of 4,096 rows instead of 2,048: cold prompts 5–7 % faster, ~2.5 GB more arena per rank. The prefill sums run in a different order, so replies differ from stock after a few words. Alone, kept states resume on a 4,096-token grid and a cached agent turn redoes ~1,000 more tokens (+24 % time to first token); with 0012 a turn resumes at its exact tail, and on the multi-turn security eval the 300 answers are those of 0012 alone. | with 0012 (recommended), else leave it out |
| 0006 | The RDMA ring's queue pairs take the port's active MTU (at most 4,096). The ring then works at Ethernet MTU 1500. | on with `--rdma` |
| 0007 | The RDMA ring picks the port's RoCE v2 IPv4 GID. Stock code uses index 5, which is empty on a usual Spark link (ours: 3), so the ring failed with `VerbsFailed`. | on with `--rdma` (`TF_DS_RDMA_GID` overrides) |
| 0008 | The tokenizer memoizes each text segment between added tokens (a chat turn tokenizes the whole conversation twice). A cached agent turn of 75K–100K tokens prepares in 0.03 s instead of 0.14 s. Same ids. | on |
| 0009 | The prompt indexer reads each row block's `vis` from a 16-byte-aligned address. At ~80K tokens of context a block is an odd number of rows (409–419), and the kit's 16-byte loads then hit an 8-byte-aligned view. | on |
| 0010 | Decode windows up to 16 rows instead of 6, so a copy from the context (an agent rewriting a file it just read) lands up to 15 tokens a round instead of 5. The sliding-window ring already keeps 16 extra slots. | on |
| 0011 | Coupled drafts: at temperature > 0 the drafter draws each draft with the target's own keyed noise at that position (the target's top-k / top-p / min-p rule on the drafter's scores, the noise scaled by λ) instead of taking its argmax. A draft lands whenever both pick the same winner from the same noise. The target's tokens never change. | off (`TF_DS_COUPLED_DRAFTS=0.5` on both ranks) |
| 0012 | Exact tail: a prompt pass also keeps its state ~130 tokens before its end (its last chunk cut there), so the next agent turn resumes there instead of up to 2,047 tokens back. The rows past a cut round differently (the MoE gate's cuBLAS GEMM picks its split of K by the chunk's row count), so replies are not bit-identical to stock. On a long-context multi-turn security eval (300 CyberMetric questions in 30 conversations over ~14K tokens of context, every turn resumed from the cache) the score is the same with and without (285 / 300, 99.3 % identical answers) and each later turn prefills 244 tokens instead of 1,434 (0.96 → 0.48 s). | `TF_DS_EXACT_TAIL=1` on rank 0 (recommended) |
| 0013 | The drafter's Markov loop learns a bias row for each token it meets (up to n rows, ~130 KB a row on each rank) instead of recomputing it from the whole Markov head (~33 MB a rank, ~140 µs) at every miss. Same drafts. | `TF_DS_MARKOV_ROWS=4096` on both ranks |
| 0014 | The coupled drafts' top-K scans 32 slices of the vocabulary in parallel, then merges them. Same candidates in the same order. | on with 0011 |
| 0015 | The host sampler skips each block of 16 logits that cannot enter its candidate list (one vector compare). Same candidates; 75 → 10 µs a row. | on |
| 0016 | Early read: a round's greedy tokens or logits go to pinned host memory right after the forward, before the drafter absorbs the round; the host waits for that copy alone and draws while the absorb runs. Same bits. | `TF_DS_EARLY_READ=1` on rank 0 |
| 0017 | The decode round's L2 prefetch settings (delay by site, pace, budget) as variables. Starting sooner and unpaced takes 0.1 ms off a round. Same bits. | `TF_DS_PF_DELAYS=0,1000,1000 TF_DS_PF_RATE=300` on both ranks |
| 0018 | The drafter pass prefetches its next weights into L2 the way the round does (stage by stage, then the head, then the next round's first layer). Same bits. | `TF_DS_DRAFT_PREFETCH=1` on both ranks |

## Results

**Stock vs this repository** (same day, same nodes, Ethernet MTU 1500). Recommended settings: every patch above (0005
included), the RDMA ring and `TF_DS_COUPLED_DRAFTS=0.5 TF_DS_MARKOV_ROWS=4096 TF_DS_TOPK_TAU=0.45 TF_DS_EARLY_READ=1 TF_DS_PF_DELAYS=0,1000,1000 TF_DS_PF_RATE=300 TF_DS_DRAFT_PREFETCH=1 TF_DS_EXACT_TAIL=1`.

| | Stock | This repository | Gain |
|---|---:|---:|---:|
| code, one stream (greedy, 384 tokens) | 91.0 tok/s | **113.4 tok/s** | +25 % |
| prose, one stream | 55.5 tok/s | **69.3 tok/s** | +25 % |
| structured, one stream | 127.0 tok/s | **145.1 tok/s** | +14 % |
| 2 streams, total | 72.8 tok/s | **91.5 tok/s** | +26 % |
| 4 streams, total | 104.7 tok/s | **130.5 tok/s** | +25 % |
| thinking at T=1 (code, prose, structured, security Q&A) | 66.7 tok/s | **85.4 tok/s** | +28 % |
| an agent rewriting a file it just read | 114.9 tok/s | **170.3 tok/s** | +48 % |
| cached agent turn, time to first token | 31.6 s | **1.88 s** | 17× |
| multi-turn chat over ~14K tokens, prefill of each later turn | 0.96 s | **0.48 s** | 2× |

With only the settings that keep the tokens (no 0005, `TF_DS_TOPK_TAU` or `TF_DS_EXACT_TAIL`) every reply is
token-for-token the stock build's (thinking at T=1: 77.0 tok/s, +15 %). The three others change
replies, not scores. Adaptive top-k (greedy; thinking on: CyberMetric 282 → 282 / 300, MMLU computer security
85 → 86 / 100, GSM8K 96 → 98 / 100, MMLU 47 → 47 / 50; thinking off: CyberMetric 477 → 477 / 500, MMLU computer
security 87 → 87, GSM8K 294 → 294 / 300, MMLU 88 → 88). The exact tail and 4,096-row chunks: 285 / 300 with and
without on the multi-turn security eval. Raw rows: `results/stock_vs_v7.jsonl`, `results/evals_topk045.json`,
`results/chain_eval.json`.

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

**Copy-heavy agent output** (patch 0010, harness, greedy; `copy.json`: rewrite a real source file with a rename, and
repeat a function with one fix):

| Request | 6-row windows | 16-row windows | Gain | Tokens a round |
|---|---:|---:|---:|---:|
| rename | 118.9 tok/s | **185.4 tok/s** | +56 % | 5.9 → 15.1 |
| fix | 110.8 tok/s | **156.6 tok/s** | +41 % | 5.5 → 11.5 |
| both | 114.9 tok/s | **170.3 tok/s** | +48 % | |

Same tokens with 6-row windows, 16-row windows and no drafts at all. Code, prose and structured requests, thinking at
T=1 and 4 parallel streams keep their speed and their tokens, and the agent replay stays 15/15 identical to stock
(`results/copy_windows.jsonl`, `results/agent_replay_local_patched_0010.json`).

**Coupled drafts** (patch 0011, harness, thinking at T=1 with the served sampling: top-p 0.95, top-k 20; one
stream, 768 tokens a request; `think4_t1.json`):

| Request | Argmax drafts | Coupled, λ 0.5 | Gain | Tokens a round |
|---|---:|---:|---:|---:|
| code | 85.9 tok/s | **90.1 tok/s** | +4.9 % | 3.67 → 3.92 |
| prose | 64.0 tok/s | 63.3 tok/s | −1.1 % | 2.23 → 2.23 |
| structured | 92.6 tok/s | **96.2 tok/s** | +3.9 % | 3.75 → 3.99 |
| security Q&A | 62.6 tok/s | **64.6 tok/s** | +3.2 % | 2.33 → 2.42 |
| all four | 73.4 tok/s | **75.1 tok/s** | +2.3 % | |

Same tokens with and without coupling. Rounds for the four requests by λ: 1,053 at 0 (the argmax), 1,025 at 0.3,
1,015–1,017 at 0.5, 1,020 at 0.7, 1,021 at 1. The drafter pass costs 0.25 ms more (3.6 → 3.9 ms), and greedy
requests skip the coupled path. Without thinking (T=1, 384 tokens each of code, prose and structured text)
coupling costs 1.7 % (86.5 → 85.0 tok/s): it pays off on thinking, where agent turns spend their tokens. With 4
streams, a shared round's 16 rows cap each stream's drafts and coupling changes nothing (108–110 against 109 tok/s);
with 2 streams, 86.5 → 87.9 tok/s.
Same tokens in every run. Raw rows: `results/coupled_drafts.jsonl`.

**Drafter and sampler** (patches 0013–0015 on top of 0011, `TF_DS_MARKOV_ROWS=4096`, harness, served sampling):

| Requests | 0011 | + 0013–0015 | Gain |
|---|---:|---:|---:|
| thinking, 4 requests, one stream | 75.1 tok/s | **76.1 tok/s** | +1.3 % |
| no thinking (T=1), 3 requests, one stream | 85.0 tok/s | **85.7 tok/s** | +0.8 % |
| thinking, 4 requests, 2 streams | 87.9 tok/s | **89.2 tok/s** | +1.5 % |

The drafter pass drops from 3.9 to 3.5–3.7 ms (the Markov loop from 0.81 to 0.50 ms at T=1) and a round's draws on
the host from 0.31 to 0.12 ms. Same tokens, same rounds. Raw rows: `results/drafter_sampler.jsonl`.

**Early read and L2 prefetch** (patches 0016–0018 on top of 0015, harness, one stream; `results/prefetch.jsonl`):

| Requests | 0015 | + 0016 | + 0017 | + 0018 |
|---|---:|---:|---:|---:|
| thinking at T=1, 4 requests | 76.1 tok/s | 76.4 | 76.6 | **77.0** |
| greedy (code, prose, structured) | 89.4 tok/s | 89.6 | | **90.0** (0017 + 0018) |

Same tokens in every run. 0017 alone, a five-row round: 38.26 → 38.13 ms; a one-row round: 22.12 → 22.00 ms.

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

**Prefill** (`tf-dsv41-lanes` harness, MTU 9000, one request at a time). Patch 0005 (recommended with 0012): the
first token is the same, then replies differ (the prefill sums run in another order):

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

| | Stock | Patched (no 0005) | Patched + 0005 | Patched + 0012 on |
|---|---:|---:|---:|---:|
| Time to first token, turns 2–5, mean | 31.6 s | **2.1 s** (15×) | 2.6 s | **1.88 s** |
| Time to first token, turns 2–5, max | 45.6 s | **2.6 s** (17×) | 3.4 s | **1.99 s** |
| Prompt tokens served from cache, turns 2–5 | 21 % | **96 %** | 95 % | 97 % |
| Wall time, 15 turns | 504 s | **148 s** (3.4×) | 146 s | 143 s |
| Replies identical to the stock build | | **15 / 15** | 0 / 15 | 1 / 15 |

Raw rows: `results/agent_replay_local_{stock,patched,patched_0005,patched_0012}.json`. With 0010–0015 and the lossless settings (`TF_DS_COUPLED_DRAFTS=0.5`, `TF_DS_MARKOV_ROWS=4096`) the replay stays 15/15 identical to stock, 2.1 s mean to first token (`results/agent_replay_local_patched_0015.json`). With 0001–0018 and the recommended settings (0005 and 0012 on), cached turns take 1.88 s mean and 2.01 s max to the first token, 138 s for the 15 turns (`results/agent_replay_local_v7.json`). Run from a remote client
over a high-latency link, each turn took 2–4 s longer: the 300–430 KB request went out on a fresh TCP connection
every turn. Keep the agents near the server, or reuse connections.

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
- Locking the GPU clocks at the 3,003 MHz maximum (`nvidia-smi -lgc 3003,3003`): under load the GB10 stays at
  2.4–2.5 GHz (prefill 2,416 MHz at 79 W, decode 2,489 MHz at 65 W) with no throttle reason reported, and prefill
  (8K / 32K / 128K: 3.57 / 13.39 / 57.25 s) and decode times do not move.
- A round-cost table measured with the ring (the served one is 8–10 ms too high): same speed, greedy, T=1 and 4
  streams. The depth rule replaces the table with measured round times after a few rounds.
- One RDMA device instead of two for the decode ring: 0.2 ms slower a five-row round. The ring's PDL off, or the
  Sinkhorn side stream folded into the main one: no gain. A ring exchange takes ~18–23 µs inside a round against
  ~10 µs alone; both ranks wait about as long, and delaying the L2 prefetch to leave DRAM to the NIC is slower.
- Pinning the host threads to the Cortex-X925 cores; the compressor's small kernels on a side stream beside the
  q GEMM: no change.
- Recalibrating the drafter's confidences, which set how many drafts a round verifies: they already match how
  often drafts land (predicted 0.80–0.86, landed 0.78–0.87, at T=0 and T=1).
- More than 4 streams or 16 rows a round: the kit has no row-invariant kernels past 16 rows
  (`MissingTritonVariant`), and batching past them would change replies.

## Build and run

Build and run as described in the upstream recipe
([bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10)),
after `git am`. Inside `nvcr.io/nvidia/pytorch:26.07-py3`:

```
zig build install native -Dtarget=aarch64-linux-gnu -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --prefix <OUT>
```

Add `TF_DS_SPILL_DIR=<dir on local NVMe>` to both ranks' environment, and mount that directory into the containers.
Recommended on both ranks: `TF_DS_COUPLED_DRAFTS=0.5 TF_DS_MARKOV_ROWS=4096 TF_DS_TOPK_TAU=0.45 TF_DS_EARLY_READ=1 TF_DS_PF_DELAYS=0,1000,1000 TF_DS_PF_RATE=300 TF_DS_DRAFT_PREFETCH=1 TF_DS_EXACT_TAIL=1`. Every one of them keeps the tokens except `TF_DS_TOPK_TAU` and `TF_DS_EXACT_TAIL` (see 0004 and 0012).

```
python3 bench/agent_replay.py http://<head>:<port> out.json
python3 bench/img_cache_check.py http://<head>:<port>
python3 bench/spill_check.py http://<head>:<port>      # on a small pool, e.g. --context 131072
```

## Next

Work in progress, in this repository:
- prompt kernels that give the same bits whatever the chunk, so 0012 keeps cached turns bit-identical to cold ones.
  The MoE gate's cuBLAS GEMM picks its split of K by the row count (always running it over 2,048 rows fixes that
  one), but other prompt kernels also change bits with the chunk.

A decode round is already at ~90 % of DRAM bandwidth and each extra verified row costs ~4 ms, mostly new experts:
what is left is in tokens per weight read (drafts, streams), not in kernels.

## Credits

DeepSeek (model, MIT), Mia-AiLab (EXL3 quant, MIT), turboderp (EXL3), ashhart (TensorFold, Apache-2.0), bertholomus
(the `dsv41` TP2 family, Apache-2.0). Patches and scripts here: Apache-2.0 (`LICENSE.md`, `NOTICE`).
