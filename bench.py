#!/usr/bin/env python3
"""Benchmark one OpenAI-compatible Qwen3.8 server: speed (decode, prefill, concurrency) and quality.

Stdlib only. Run with `python3 -I`. Results go to --out as JSON, including every raw generation,
so two engines can be compared item by item afterwards.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from http.client import HTTPConnection
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))

DECODE_PROMPTS = [
    "Explain how a transformer language model generates text, one token at a time, for a curious high-school student.",
    "Write an essay on why local inference on a single workstation matters for privacy and cost.",
    "Describe the trade-offs between speculative decoding and plain autoregressive decoding.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
]
CODE_PROMPT = "Write a Python quicksort with a docstring and tests."
FILLER = (
    "the river bends past old mills where children once fished and farmers sold apples at dawn while "
    "engineers measured the tides and clerks recorded every shipment in careful ledgers that nobody read "
    "again until the archive flooded and the paper turned to pulp along the quiet stone embankment "
).split()


class Client:
    def __init__(self, base, model):
        self.url = urlparse(base.rstrip("/") + "/chat/completions")
        self.model = model

    def stream(self, messages, max_tokens):
        """One streaming chat request. Returns timing marks, the text and the server's usage block."""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        conn = HTTPConnection(self.url.hostname, self.url.port, timeout=1800)
        t0 = time.perf_counter()
        conn.request("POST", self.url.path, body=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
        parts, usage, t_first, t_last = [], None, None, None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    parts.append(piece)
        t_end = time.perf_counter()
        conn.close()
        return {
            "text": "".join(parts), "t0": t0, "t_first": t_first, "t_last": t_last or t_end,
            "t_end": t_end, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
        }


def nonce_prefix():
    # Unique prefix per request, so no prompt-cache hit can flatter TTFT.
    return f"[request {uuid.uuid4().hex}] "


def decode_speed(r):
    n = r["completion_tokens"] or 0
    span = r["t_last"] - r["t_first"]
    return (n - 1) / span if n > 1 and span > 0 else None


def summary(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals.sort()
    q = statistics.quantiles(vals, n=4) if len(vals) >= 2 else [vals[0]] * 3
    return {"median": statistics.median(vals), "q1": q[0], "q3": q[2], "min": vals[0], "max": vals[-1], "n": len(vals)}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [max(0.0, c - h), min(1.0, c + h)]


# ---------- speed ----------

def warmup(cli):
    for _ in range(2):
        cli.stream([{"role": "user", "content": "Say hello."}], 32)


def decode_concurrency(cli, levels, repeats, max_tokens):
    out = {}
    for k in levels:
        per_req, aggregate, ttft = [], [], []
        for rep in range(repeats):
            prompts = [nonce_prefix() + DECODE_PROMPTS[i % len(DECODE_PROMPTS)] for i in range(k)]
            with cf.ThreadPoolExecutor(max_workers=k) as ex:
                rs = list(ex.map(lambda p: cli.stream([{"role": "user", "content": p}], max_tokens), prompts))
            per_req += [decode_speed(r) for r in rs]
            ttft += [r["t_first"] - r["t0"] for r in rs]
            total = sum(r["completion_tokens"] for r in rs)
            wall = max(r["t_last"] for r in rs) - min(r["t_first"] for r in rs)
            aggregate.append(total / wall)
            print(f"  decode c={k} rep={rep + 1}: aggregate {total / wall:.1f} tok/s, "
                  f"per-request median {statistics.median(decode_speed(r) for r in rs):.1f}", flush=True)
        out[str(k)] = {"aggregate_tok_s": summary(aggregate), "per_request_tok_s": summary(per_req),
                       "ttft_s": summary(ttft)}
    return out


def code_speed(cli, repeats, max_tokens):
    speeds, ttfts = [], []
    for _ in range(repeats):
        r = cli.stream([{"role": "user", "content": nonce_prefix() + CODE_PROMPT}], max_tokens)
        speeds.append(decode_speed(r))
        ttfts.append(r["t_first"] - r["t0"])
    return {"per_request_tok_s": summary(speeds), "ttft_s": summary(ttfts)}


def prefill(cli, sizes, repeats):
    out = {}
    for size in sizes:
        rates, ttfts, measured = [], [], []
        for rep in range(repeats):
            rng = random.Random(f"{size}-{rep}-{uuid.uuid4().hex}")
            words = [rng.choice(FILLER) for _ in range(int(size / 1.3))]
            prompt = "Summarize the following text in one sentence.\n\n" + " ".join(words)
            r = cli.stream([{"role": "user", "content": prompt}], 1)
            ttft = r["t_first"] - r["t0"]
            rates.append(r["prompt_tokens"] / ttft)
            ttfts.append(ttft)
            measured.append(r["prompt_tokens"])
            print(f"  prefill ~{size}: {r['prompt_tokens']} tokens, TTFT {ttft:.2f}s, "
                  f"{r['prompt_tokens'] / ttft:.0f} tok/s", flush=True)
        out[str(size)] = {"prompt_tokens": measured, "prefill_tok_s": summary(rates), "ttft_s": summary(ttfts)}
    return out


# ---------- quality ----------

def load_gsm8k(n):
    rows = json.load(open(os.path.join(HERE, "data", "gsm8k_test100.json")))["rows"][:n]
    return [r["row"] for r in rows]


def load_humaneval(n):
    rows = json.load(open(os.path.join(HERE, "data", "humaneval20.json")))["rows"][:n]
    return [r["row"] for r in rows]


def last_number(text):
    m = re.findall(r"ANSWER:\s*\**\s*\$?(-?[\d,]*\.?\d+)", text)
    cand = m[-1] if m else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
    if cand is None:
        return None
    try:
        return float(cand.replace(",", ""))
    except ValueError:
        return None


def gsm8k(cli, n, max_tokens):
    items, correct = [], 0
    for i, row in enumerate(load_gsm8k(n)):
        gold = float(row["answer"].split("####")[-1].strip().replace(",", ""))
        prompt = (row["question"] + "\n\nReason step by step. End with a final line of the form "
                  "'ANSWER: <number>'.")
        r = cli.stream([{"role": "user", "content": prompt}], max_tokens)
        pred = last_number(r["text"])
        ok = pred is not None and abs(pred - gold) < 1e-6
        correct += ok
        items.append({"idx": i, "gold": gold, "pred": pred, "ok": ok, "text": r["text"],
                      "completion_tokens": r["completion_tokens"]})
        if (i + 1) % 10 == 0:
            print(f"  gsm8k {i + 1}/{n}: {correct} correct so far", flush=True)
    return {"n": n, "correct": correct, "accuracy": correct / n, "wilson95": wilson(correct, n), "items": items}


def extract_code(text, entry_point, prompt):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    body = blocks[-1] if blocks else text
    if f"def {entry_point}" in body:
        return body
    return prompt + body  # the model returned only the body


def humaneval(cli, n, max_tokens):
    items, passed = [], 0
    with tempfile.TemporaryDirectory(prefix="he-") as work:
        for i, row in enumerate(load_humaneval(n)):
            prompt = ("Complete the following Python function. Return the full function in one ```python block.\n\n"
                      + row["prompt"])
            r = cli.stream([{"role": "user", "content": prompt}], max_tokens)
            program = extract_code(r["text"], row["entry_point"], row["prompt"])
            program += "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n"
            path = os.path.join(work, f"p{i}.py")
            with open(path, "w") as f:
                f.write(program)
            try:
                rc = subprocess.run([sys.executable, "-I", path], cwd=work, capture_output=True, timeout=10).returncode
            except subprocess.TimeoutExpired:
                rc = -1
            ok = rc == 0
            passed += ok
            items.append({"task": row["task_id"], "ok": ok, "text": r["text"],
                          "completion_tokens": r["completion_tokens"]})
            print(f"  humaneval {i + 1}/{n}: {row['task_id']} {'PASS' if ok else 'fail'}", flush=True)
    return {"n": n, "passed": passed, "pass_at_1": passed / n, "wilson95": wilson(passed, n), "items": items}


def memory_snapshot(container):
    snap = {}
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        snap["container_mem"] = out
    except (subprocess.SubprocessError, OSError):
        snap["container_mem"] = None
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            snap["mem_available_gib"] = round(int(line.split()[1]) / 1048576, 2)
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--container", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--n-gsm8k", type=int, default=100)
    ap.add_argument("--n-humaneval", type=int, default=20)
    ap.add_argument("--skip-quality", action="store_true")
    args = ap.parse_args()

    cli = Client(args.base, args.model)
    result = {"engine": args.engine, "model": args.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "settings": {"temperature": 0, "top_p": 1, "thinking": False, "max_tokens_decode": 256,
                           "max_tokens_gsm8k": 1024, "max_tokens_humaneval": 1024, "repeats": args.repeats},
              "memory_before": memory_snapshot(args.container)}
    print(f"[{args.engine}] warm-up", flush=True)
    warmup(cli)
    print(f"[{args.engine}] decode speed by concurrency", flush=True)
    result["decode"] = decode_concurrency(cli, [int(x) for x in args.levels.split(",")], args.repeats, 256)
    print(f"[{args.engine}] code decode", flush=True)
    result["code_decode"] = code_speed(cli, args.repeats, 256)
    print(f"[{args.engine}] prefill", flush=True)
    result["prefill"] = prefill(cli, [2048, 8192, 16384], args.repeats)
    result["memory_after_speed"] = memory_snapshot(args.container)
    if not args.skip_quality:
        print(f"[{args.engine}] GSM8K (greedy, sequential)", flush=True)
        result["gsm8k"] = gsm8k(cli, args.n_gsm8k, 1024)
        print(f"[{args.engine}] HumanEval (greedy, sequential)", flush=True)
        result["humaneval"] = humaneval(cli, args.n_humaneval, 1024)
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    result["memory_after"] = memory_snapshot(args.container)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{args.engine}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
