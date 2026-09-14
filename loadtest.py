#!/usr/bin/env python3
"""Concurrency sweep against the vLLM API while sampling RoCE port counters on both Sparks.

Usage: ./loadtest.py [--max-tokens N] [--thinking] [LEVEL ...]    (default levels: 1 2 4 8)

Settings are resolved exactly like the shell scripts (cluster.env or CLUSTER_ENV, plus
HEAD_HOST/WORKER_HOST overrides from the environment). Standard library only, so it runs
with the macOS system python3.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOPICS = ["volcanoes", "the history of the printing press", "how TCP congestion control works",
          "coral reef ecosystems", "the Apollo guidance computer", "sourdough fermentation",
          "how compilers optimize loops", "the economics of shipping containers"]
# port_xmit_data / port_rcv_data count 4-octet units.
SNAP = ("for d in /sys/class/infiniband/*; do c=$d/ports/1/counters; "
        "echo $(basename $d) $(cat $c/port_xmit_data) $(cat $c/port_rcv_data); done")


def load_config():
    """Source lib.sh in bash so config selection and overrides match the scripts."""
    script = 'source "$1/lib.sh" && printf "%s\\n" "$HEAD_HOST" "$WORKER_HOST" "$API_PORT" "$MODEL_ID"'
    r = subprocess.run(["bash", "-c", script, "loadtest", str(ROOT)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr.strip() or "failed to load cluster config")
    head, worker, port, model_id = r.stdout.splitlines()[-4:]
    return head, worker, port, model_id


def snapshot(hosts):
    out = {}
    for host in hosts:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, SNAP],
                           capture_output=True, text=True, check=True)
        for line in r.stdout.splitlines():
            if line.strip():
                dev, tx, rx = line.split()
                out[(host, dev)] = (int(tx), int(rx))
    return out


def long_document(approx_tokens, seed):
    """Unique filler text (~4 chars per token) so requests don't share a cached prefix."""
    lines = [f"Document {seed}. Maintenance log for cluster site {seed}."]
    size = len(lines[0])
    k = 0
    while size < approx_tokens * 4:
        line = (f"Entry {k}: node {seed}-{k % 97} reported temperature {40 + (k * 7) % 45} C, "
                f"fan {1200 + (k * 13) % 3000} rpm, link {k % 4} at {100 + (k * 3) % 100} Gbit/s.")
        lines.append(line)
        size += len(line) + 1
        k += 1
    return "\n".join(lines)


def one_request(args, url, model, i, n, results):
    task = f"Write a detailed ~500 word explainer about {TOPICS[i % len(TOPICS)]}. (req {n}-{i})"
    if args.prompt_tokens:
        task = (f"{long_document(args.prompt_tokens, f'{n}-{i}-{time.time_ns()}')}\n\n"
                f"Summarize the maintenance log above in about 300 words, then list the three hottest entries.")
    body = {"model": model, "max_tokens": args.max_tokens, "temperature": 0.7, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": args.thinking},
            "messages": [{"role": "user", "content": task}]}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    tokens = 0
    prompt_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                delta = d["choices"][0].get("delta", {}) if d.get("choices") else {}
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")):
                    first = time.time()
                if d.get("usage"):
                    tokens = d["usage"]["completion_tokens"]
                    prompt_tokens = d["usage"]["prompt_tokens"]
        end = time.time()
        results[i] = dict(ok=True, ttft=(first or end) - t0, tokens=tokens, prompt=prompt_tokens,
                          decode=(tokens - 1) / max(end - (first or t0), 1e-6))
    except Exception as e:  # noqa: BLE001 - report every failure in the table
        results[i] = dict(ok=False, err=str(e))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("levels", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="send a unique ~N-token document in every request (0 = short prompt)")
    parser.add_argument("--thinking", action="store_true", help="leave DeepSeek thinking on")
    args = parser.parse_args()

    head, worker, port, model = load_config()
    hosts = [head, worker]
    url = f"http://{head}:{port}/v1/chat/completions"
    print(f"{model} @ {url}, max_tokens={args.max_tokens}, prompt_tokens~{args.prompt_tokens or 'short'}, "
          f"thinking={args.thinking}\n")

    total_before = snapshot(hosts)
    print(f"{'conc':>4} {'ok':>4} {'wall s':>7} {'agg tok/s':>9} {'per-req tok/s':>13} {'TTFT avg s':>10} "
          f"{'TTFT max s':>10} {'prompt tok':>10} {'prefill tok/s':>13}")
    for n in args.levels:
        results = [None] * n
        threads = [threading.Thread(target=one_request, args=(args, url, model, i, n, results)) for i in range(n)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        ok = [r for r in results if r and r["ok"]]
        toks = sum(r["tokens"] for r in ok)
        per_req = sum(r["decode"] for r in ok) / max(len(ok), 1)
        ttfts = [r["ttft"] for r in ok] or [0]
        prompts = sum(r["prompt"] for r in ok)
        # Prompts are processed together, so aggregate prefill = all prompt tokens / slowest TTFT.
        prefill = prompts / max(max(ttfts), 1e-6)
        print(f"{n:>4} {len(ok):>4} {wall:>7.1f} {toks / wall:>9.1f} {per_req:>13.1f} "
              f"{sum(ttfts) / len(ttfts):>10.2f} {max(ttfts):>10.2f} "
              f"{prompts // max(len(ok), 1):>10} {prefill:>13.0f}")
        for r in results:
            if r and not r["ok"]:
                print("     error:", r["err"])
    total_after = snapshot(hosts)

    print("\nRoCE traffic during the sweep (GB). f0 devices = QSFP port 0, f1 devices = QSFP port 1:")
    for host in hosts:
        for dev in sorted(d for (h, d) in total_after if h == host):
            tx = (total_after[(host, dev)][0] - total_before[(host, dev)][0]) * 4 / 1e9
            rx = (total_after[(host, dev)][1] - total_before[(host, dev)][1]) * 4 / 1e9
            print(f"  {host:<11} {dev:<14} tx {tx:8.3f}  rx {rx:8.3f}")


if __name__ == "__main__":
    main()
