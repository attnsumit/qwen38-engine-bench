#!/usr/bin/env python3
"""Build the comparison from the two result files: a shareable HTML card and a methods/stats summary."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "results", "results-tensorfold.json")))
B = json.load(open(os.path.join(HERE, "results", "results-flare.json")))
OUT_HTML = os.path.join(HERE, "report", "qwen38-engine-showdown.html")
OUT_MD = os.path.join(HERE, "report", "METHODS-AND-RESULTS.md")


def med(d):
    return d["median"] if d else None


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def pct_gain(a, b):
    """How much faster engine a is than engine b, in percent (b is the baseline)."""
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def mcnemar_exact(a_ok, b_ok):
    """Exact McNemar test on paired per-item outcomes (two-sided, binomial)."""
    b01 = sum(1 for x, y in zip(a_ok, b_ok) if (not x) and y)
    b10 = sum(1 for x, y in zip(a_ok, b_ok) if x and (not y))
    n = b01 + b10
    if n == 0:
        return 1.0, b01, b10
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p), b01, b10


def main():
    dA, dB = A["decode"], B["decode"]
    levels = sorted(dA.keys(), key=int)
    gsA, gsB = A["gsm8k"], B["gsm8k"]
    heA, heB = A["humaneval"], B["humaneval"]

    # Paired per-item comparison: same questions, greedy decoding, so the outcomes should agree
    a_ok = [i["ok"] for i in gsA["items"]]
    b_ok = [i["ok"] for i in gsB["items"]]
    p_gsm, gsm_b01, gsm_b10 = mcnemar_exact(a_ok, b_ok)
    same_text_gsm = sum(1 for x, y in zip(gsA["items"], gsB["items"]) if x["text"].strip() == y["text"].strip())
    same_text_he = sum(1 for x, y in zip(heA["items"], heB["items"]) if x["text"].strip() == y["text"].strip())

    speed_rows = []
    for k in levels:
        a, b = dA[k], dB[k]
        speed_rows.append({
            "c": int(k),
            "agg_a": med(a["aggregate_tok_s"]), "agg_b": med(b["aggregate_tok_s"]),
            "per_a": med(a["per_request_tok_s"]), "per_b": med(b["per_request_tok_s"]),
            "ttft_a": med(a["ttft_s"]), "ttft_b": med(b["ttft_s"]),
        })
    prefill_rows = []
    for k in sorted(A["prefill"].keys(), key=int):
        prefill_rows.append({
            "tokens": round(sum(A["prefill"][k]["prompt_tokens"]) / len(A["prefill"][k]["prompt_tokens"])),
            "a": med(A["prefill"][k]["prefill_tok_s"]), "b": med(B["prefill"][k]["prefill_tok_s"]),
            "ttft_a": med(A["prefill"][k]["ttft_s"]), "ttft_b": med(B["prefill"][k]["ttft_s"]),
        })

    headline_gain = pct_gain(speed_rows[0]["per_b"], speed_rows[0]["per_a"])
    gsm_a, gsm_b = gsA["accuracy"] * 100, gsB["accuracy"] * 100

    data = {
        "a": {"name": "MiaAI TensorFold (Zig engine)", "weights": "NVIDIA NVFP4 FP4 checkpoint"},
        "b": {"name": "NVIDIA Flare", "weights": "Flare-compiled package (same weights)"},
        "speed": speed_rows, "prefill": prefill_rows,
        "gsm8k": {"a": [gsA["correct"], gsA["n"]], "b": [gsB["correct"], gsB["n"]],
                  "a_ci": gsA["wilson95"], "b_ci": gsB["wilson95"], "p": p_gsm, "b01": gsm_b01, "b10": gsm_b10,
                  "same_text": same_text_gsm},
        "humaneval": {"a": [heA["passed"], heA["n"]], "b": [heB["passed"], heB["n"]],
                      "a_ci": heA["wilson95"], "b_ci": heB["wilson95"], "same_text": same_text_he},
        "code_decode": {"a": med(A["code_decode"]["per_request_tok_s"]), "b": med(B["code_decode"]["per_request_tok_s"])},
        "headline_gain": headline_gain,
        "memory": {"a": A.get("memory_after_speed", {}).get("container_mem"),
                   "b": B.get("memory_after_speed", {}).get("container_mem")},
        "gsm8k_acc": (gsm_a, gsm_b),
    }
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(os.path.join(HERE, "report", "data.json"), "w") as f:
        json.dump(data, f, indent=1)

    # Methods and full numbers, for the reader who wants to check the claim
    lines = []
    lines.append("# Qwen3.8-Flash-Next on one DGX Spark: TensorFold vs NVIDIA Flare\n")
    lines.append(f"Run dates: TensorFold {A['started']} to {A['finished']}; Flare {B['started']} to {B['finished']}.\n")
    lines.append("## Setup\n")
    lines.append("- Hardware: one NVIDIA DGX Spark (GB10, 121 GiB unified memory), driver 580.178.04.")
    lines.append("- One engine loaded at a time, both on 127.0.0.1:8888, the same OpenAI-compatible chat endpoint.")
    lines.append("- Settings for both: greedy decoding (temperature 0, top-k 1), thinking off, MTP speculative drafts on, "
                 "8 concurrent slots, vision off, text only.")
    lines.append("- Speed: 256-token decode with fixed prompts; concurrency 1, 2, 4, 8; 3 repeats each (median "
                 "reported); a nonce prefix on every request defeats prompt caches. Prefill at ~2k, ~8k and ~16k tokens, "
                 "3 repeats, cold prompts.")
    lines.append("- Quality: GSM8K test set, first 100 questions (openai/gsm8k, `main`), greedy, sequential, up to 1024 "
                 "tokens, final `ANSWER:` line. HumanEval, first 20 tasks (openai_humaneval), greedy, sequential, "
                 "executed in a separate Python process with a 10 s timeout.")
    lines.append("- Caveat: the page cache could not be flushed (no passwordless sudo), so the engines' cold-load "
                 "behaviour is not part of this comparison. Both use the same weights (the Flare package was "
                 "built from a repo whose shards match the NVIDIA checkpoint byte-for-byte).\n")
    lines.append("## Speed (median of 3)\n")
    lines.append("| Concurrent | TensorFold aggregate | Flare aggregate | TensorFold per request | Flare per request | TensorFold TTFT | Flare TTFT |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in speed_rows:
        lines.append(f"| {r['c']} | {fmt(r['agg_a'])} | {fmt(r['agg_b'])} | {fmt(r['per_a'])} | {fmt(r['per_b'])} | "
                     f"{fmt(r['ttft_a'] * 1000, 0)} ms | {fmt(r['ttft_b'] * 1000, 0)} ms |")
    lines.append("")
    lines.append(f"Code decode, single request (median): TensorFold {fmt(data['code_decode']['a'])} tok/s, "
                 f"Flare {fmt(data['code_decode']['b'])} tok/s.\n")
    lines.append("## Prefill (cold prompts, median of 3)\n")
    lines.append("| Prompt tokens | TensorFold prefill | Flare prefill | TensorFold TTFT | Flare TTFT |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for r in prefill_rows:
        lines.append(f"| {r['tokens']:,} | {fmt(r['a'], 0)} tok/s | {fmt(r['b'], 0)} tok/s | "
                     f"{fmt(r['ttft_a'], 2)} s | {fmt(r['ttft_b'], 2)} s |")
    lines.append("")
    lines.append("## Quality\n")
    lines.append(f"- GSM8K (n=100): TensorFold {gsA['correct']}/100 ({gsm_a:.1f}%, Wilson 95% CI "
                 f"{gsA['wilson95'][0]*100:.1f}-{gsA['wilson95'][1]*100:.1f}); Flare {gsB['correct']}/100 "
                 f"({gsm_b:.1f}%, CI {gsB['wilson95'][0]*100:.1f}-{gsB['wilson95'][1]*100:.1f}).")
    lines.append(f"  - Paired exact McNemar test: {gsm_b01} items only Flare solved, {gsm_b10} only TensorFold solved, "
                 f"p = {p_gsm:.3f}.")
    lines.append(f"  - Identical full text in {same_text_gsm}/100 items.")
    lines.append(f"- HumanEval (n=20): TensorFold {heA['passed']}/20 ({heA['pass_at_1']*100:.0f}%); Flare "
                 f"{heB['passed']}/20 ({heB['pass_at_1']*100:.0f}%). Identical text in {same_text_he}/20 tasks.")
    lines.append("")
    lines.append("## Memory\n")
    lines.append(f"- Container memory after the speed runs: TensorFold {data['memory']['a']}; Flare {data['memory']['b']}.\n")
    lines.append("## Reading the numbers\n")
    lines.append("- Speed and accuracy share one caveat: n=100 and n=20 are small samples. A difference under about "
                 "10 points on GSM8K or 20 points on HumanEval is within noise.")
    lines.append("- Greedy decoding is deterministic per engine, but two engines do not compute identical floating-point "
                 "results, so some divergence is expected even with the same weights.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", OUT_MD, file=sys.stderr)
    print("wrote", os.path.join(HERE, "report", "data.json"), file=sys.stderr)


if __name__ == "__main__":
    main()
