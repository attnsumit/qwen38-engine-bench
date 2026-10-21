# Qwen3.8-Flash-Next on one DGX Spark: TensorFold vs NVIDIA Flare

Run dates: TensorFold 2026-10-21T16:09:45 to 2026-10-21T16:25:06; Flare 2026-10-21T16:25:45 to 2026-10-21T16:46:41.

## Setup

- Hardware: one NVIDIA DGX Spark (GB10, 121 GiB unified memory), driver 580.178.04.
- One engine loaded at a time, both on 127.0.0.1:8888, the same OpenAI-compatible chat endpoint.
- Settings for both: greedy decoding (temperature 0, top-k 1), thinking off, MTP speculative drafts on, 8 concurrent slots, vision off, text only.
- Speed: 256-token decode with fixed prompts; concurrency 1, 2, 4, 8; 3 repeats each (median reported); a nonce prefix on every request defeats prompt caches. Prefill at ~2k, ~8k and ~16k tokens, 3 repeats, cold prompts.
- Quality: GSM8K test set, first 100 questions (openai/gsm8k, `main`), greedy, sequential, up to 1024 tokens, final `ANSWER:` line. HumanEval, first 20 tasks (openai_humaneval), greedy, sequential, executed in a separate Python process with a 10 s timeout.
- Caveat: the page cache could not be flushed (no passwordless sudo), so the engines' cold-load behaviour is not part of this comparison. Both use the same weights (the Flare package was built from a repo whose shards match the NVIDIA checkpoint byte-for-byte).

## Speed (median of 3)

| Concurrent | TensorFold aggregate | Flare aggregate | TensorFold per request | Flare per request | TensorFold TTFT | Flare TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.7 | 39.3 | 38.5 | 39.1 | 214 ms | 251 ms |
| 2 | 56.1 | 48.7 | 29.6 | 24.6 | 291 ms | 500 ms |
| 4 | 77.8 | 66.8 | 21.2 | 17.4 | 363 ms | 995 ms |
| 8 | 115.5 | 76.0 | 16.0 | 9.7 | 480 ms | 1,966 ms |

Code decode, single request (median): TensorFold 64.0 tok/s, Flare 35.2 tok/s.

## Prefill (cold prompts, median of 3)

| Prompt tokens | TensorFold prefill | Flare prefill | TensorFold TTFT | Flare TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 1,786 | 2,232 tok/s | 1,474 tok/s | 0.80 s | 1.22 s |
| 7,082 | 2,175 tok/s | 1,867 tok/s | 3.25 s | 3.80 s |
| 14,196 | 2,304 tok/s | 1,958 tok/s | 6.16 s | 7.23 s |

## Quality

- GSM8K (n=100): TensorFold 97/100 (97.0%, Wilson 95% CI 91.5-99.0); Flare 97/100 (97.0%, CI 91.5-99.0).
  - Paired exact McNemar test: 1 items only Flare solved, 1 only TensorFold solved, p = 1.000.
  - Identical full text in 0/100 items.
- HumanEval (n=20): TensorFold 20/20 (100%); Flare 20/20 (100%). Identical text in 7/20 tasks.

## Memory

- Container memory after the speed runs: TensorFold 1.39GiB / 121.7GiB; Flare 2.73GiB / 121.7GiB.

## Reading the numbers

- Speed and accuracy share one caveat: n=100 and n=20 are small samples. A difference under about 10 points on GSM8K or 20 points on HumanEval is within noise.
- Greedy decoding is deterministic per engine, but two engines do not compute identical floating-point results, so some divergence is expected even with the same weights.
