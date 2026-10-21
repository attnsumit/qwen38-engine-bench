# Qwen3.8-Flash-Next on one DGX Spark: TensorFold vs NVIDIA Flare

A small, reproducible benchmark of two local serving engines for the same Qwen3.8-Flash-Next weights, run on one
NVIDIA DGX Spark (GB10, 121 GiB unified memory).

- **MiaAI TensorFold** (Zig engine, `nvidia/Qwen3.8-Flash-Next-FP4-Local` NVFP4 checkpoint)
- **NVIDIA Flare** (`local-inference-lab/Qwen3.8-Flash-Next-FLARE`, a compiled package of the same weights)

Only one engine is loaded at a time, both on the same OpenAI-compatible endpoint (`127.0.0.1:8888`).

## Results

| | TensorFold | Flare |
|---|---:|---:|
| Decode, 1 request (tok/s) | 38.7 | 39.3 |
| Decode, 8 requests, aggregate (tok/s) | **115.5** | 76.0 |
| TTFT, 8 requests | **480 ms** | 1,966 ms |
| Prefill, 1.8k / 7.1k / 14.2k prompt tokens (tok/s) | **2,232 / 2,175 / 2,304** | 1,474 / 1,867 / 1,958 |
| Code decode, 1 request (tok/s) | **64.0** | 35.2 |
| GSM8K, first 100 questions | 97/100 | 97/100 |
| HumanEval, first 20 tasks | 20/20 | 20/20 |

Speed numbers are medians of 3 repeats. Accuracy is the same; the two engines disagree on 2 of 100 GSM8K items
(paired exact McNemar test, p = 1.0). Full tables and methods: [`report/METHODS-AND-RESULTS.md`](report/METHODS-AND-RESULTS.md).
Shareable card: [`report/qwen38-engine-showdown.png`](report/qwen38-engine-showdown.png).

## Method

- **Settings for both engines:** greedy decoding (temperature 0, top-p 1), thinking off, MTP speculative drafts on,
  8 concurrent slots, text only (vision off).
- **Decode:** 256-token replies, concurrency 1, 2, 4, 8, plus one code prompt.
- **Prefill:** cold prompts of about 2k, 8k and 16k tokens.
- **Prompt caches defeated:** every request gets a unique nonce prefix, so a cache hit cannot flatter TTFT.
- **Quality:** GSM8K test set (`openai/gsm8k`, `main`, first 100), final `ANSWER:` line parsed; HumanEval
  (`openai/openai_humaneval`, first 20), code run in a separate Python process with a 10 s timeout.

## Limitations

- **Small samples.** 100 GSM8K items and 20 HumanEval tasks. Accuracy gaps under about 10 points are within noise.
- **One machine, one checkpoint family.** These are not general claims about either engine.
- **Cold-load time not measured.** The page cache could not be flushed (no passwordless sudo).
- **Single-stream decode was below the TensorFold README's own figure** (46–55 tok/s there, 38–40 tok/s here).
  The numbers here are what this run measured.

## Reproduce

Requirements: Python 3.10+ (standard library only, no `pip install`), Docker with the NVIDIA runtime, and the
checkpoint in the Hugging Face cache.

1. Start one engine and make sure it answers on `127.0.0.1:8888`.
   - TensorFold: `./start.sh` in the TensorFold recipe, with `THINKING=0 TEMPERATURE=0 TOP_P=1 TOP_K=1 PARALLEL=8 VISION=0 HOST=127.0.0.1`.
   - Flare: `docker compose up -d` using a compose file that serves the FLARE package on port 8888.
2. Run the benchmark:

   ```sh
   python3 -I bench.py --engine tensorfold-nvfp4 --container qwen38-flash-next-tf --out results/results-tensorfold.json
   ```

   Use `--container` to name the engine's Docker container, so memory is sampled. `--skip-quality` runs speed only.
3. Stop that engine, start the other one, and repeat with its own `--engine`, `--container` and `--out`.
4. Build the comparison and report:

   ```sh
   python3 -I make_report.py
   ```

   This reads `results/results-*.json` and writes `report/data.json` and `report/METHODS-AND-RESULTS.md`.

The HTML card in `report/` is hand-laid from `report/data.json`. The PNG was rendered from it with headless Chromium.

## Files

| Path | What it is |
|---|---|
| `bench.py` | Speed and quality harness (stdlib only). Writes raw results, including every generation. |
| `make_report.py` | Reads both result files, computes the paired statistics and writes the methods report. |
| `data/gsm8k_test100.json`, `data/humaneval20.json` | Public benchmark rows, first 100 GSM8K test and first 20 HumanEval. |
| `results/results-tensorfold.json`, `results/results-flare.json` | Raw measurements and generations for each engine. |
| `report/` | Methods and results, the card (HTML and PNG), and `data.json`. |
| `logs/` | Run logs for both benchmark runs and the server startup logs. |

## Safety notes

- `bench.py` runs model-generated code from HumanEval in a subprocess with a timeout. Run it in a disposable
  environment. It is not a sandbox.
- The servers in this setup have no API key. Bind them to loopback (`127.0.0.1`) only.
- The datasets are from the Hugging Face datasets server (GSM8K and HumanEval; check their dataset cards for terms). The model weights
  are not included here. The NVIDIA and Qwen model terms, and the Flare package terms, still apply.
