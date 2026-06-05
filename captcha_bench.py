#!/usr/bin/env python3
"""Benchmark LLM captcha-reading accuracy against classificadas/ images.

Usage:
    python3 captcha_bench.py [--model MODEL] [--server URL] [--dir DIR]
                             [--limit N] [--concurrency N] [--output FILE]
                             [--sample N] [--seed S]

The images live in classificadas/*.png where the filename (minus .png) is the
ground-truth captcha text.  Comparison is case-insensitive.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Levenshtein distance (no external deps)
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Classic DP Levenshtein distance."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]

# ---------------------------------------------------------------------------
# LLM query via llama.cpp / OpenAI-compatible chat completions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a captcha reader. You will be given an image containing a captcha. "
    "Reply with ONLY the captcha characters, nothing else. No punctuation, no explanation."
)

USER_PROMPT = "Read this captcha image and reply with only the characters shown."


def ask_llm(
    image_b64: str,
    server: str,
    model: str,
    temperature: float | None = None,
    max_tokens: int = 16384,
    enable_thinking: bool | None = None,
    max_retries: int = 3,
    timeout: int = 120,
    debug: bool = False,
) -> dict:
    """Send a single captcha image to the LLM.
    Returns dict with 'content', 'reasoning', 'raw_response', 'error'."""
    url = f"{server}/v1/chat/completions"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            },
        ],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            raw = content or reasoning  # for debugging
            if debug:
                print(f"  [DEBUG] content={content!r}")
                print(f"  [DEBUG] reasoning={reasoning[:500]!r}{'...' if len(reasoning) > 500 else ''}")
                print(f"  [DEBUG] finish_reason={data['choices'][0].get('finish_reason')!r}")
                print(f"  [DEBUG] usage={data.get('usage', {})}")
            # If content is empty, the model didn't finish thinking — count as failure
            if not content:
                return {"content": "", "reasoning": reasoning, "raw_response": raw, "error": None}
            return {"content": content, "reasoning": reasoning, "raw_response": raw, "error": None}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"content": "", "reasoning": "", "raw_response": "", "error": str(e)}
            time.sleep(2 ** attempt)

    return {"content": "", "reasoning": "", "raw_response": "", "error": "max retries"}


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_bench(args):
    img_dir = Path(args.dir)
    images = sorted(img_dir.glob("*.png"))

    if not images:
        print(f"No .png files found in {img_dir}", file=sys.stderr)
        sys.exit(1)

    if args.sample:
        import random
        rng = random.Random(args.seed)
        images = rng.sample(images, min(args.sample, len(images)))

    if args.limit:
        images = images[: args.limit]

    total = len(images)
    print(f"Benchmarking {total} captchas …")
    print(f"  Server : {args.server}")
    thinking_status = "enabled" if args.enable_thinking is True else "disabled" if args.enable_thinking is False else "server-default"
    print(f"  Model  : {args.model}")
    print(f"  Dir    : {img_dir}")
    print(f"  Concur : {args.concurrency}")
    print(f"  Thinking: {thinking_status}")
    print()

    results = []
    done = 0

    def process_one(img_path: Path):
        gt = img_path.stem  # ground truth from filename
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        llm = ask_llm(b64, args.server, args.model, args.temperature, args.max_tokens, args.enable_thinking, debug=args.debug)
        response = llm["content"]
        # Strip thinking tags if present (Qwen3 style)
        clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Keep only alphanumeric characters (captchas are alphanumeric)
        predicted = re.sub(r"[^a-zA-Z0-9]", "", clean)
        # Sanity: if predicted is much longer than ground truth, it's likely
        # the model produced prose instead of just the captcha.  Try to salvage
        # by taking the first len(gt) alphanumeric chars, or mark as empty.
        if len(predicted) > len(gt) + 2:
            predicted = predicted[: len(gt)] if len(predicted) >= len(gt) else ""
        gt_lower = gt.lower()
        pred_lower = predicted.lower()
        dist = levenshtein(gt_lower, pred_lower)
        correct = dist == 0
        return {
            "file": img_path.name,
            "ground_truth": gt,
            "predicted": predicted,
            "levenshtein": dist,
            "correct": correct,
            "error": llm.get("error"),
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process_one, img): img for img in images}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            status = "✓" if result["correct"] else "✗"
            if done % 10 == 0 or done == total or args.verbose:
                print(
                    f"  [{done}/{total}] {status}  gt={result['ground_truth']!r}  "
                    f"pred={result['predicted']!r}  lev={result['levenshtein']}"
                )

    # Sort results by filename for determinism
    results.sort(key=lambda r: r["file"])

    # ---------- Accuracy stats ----------
    total_correct = sum(1 for r in results if r["correct"])
    overall_accuracy = total_correct / len(results) if results else 0.0

    # Group by Levenshtein distance
    by_lev: dict[int, list] = {}
    for r in results:
        by_lev.setdefault(r["levenshtein"], []).append(r)

    accuracy_by_lev = {}
    for dist in sorted(by_lev):
        group = by_lev[dist]
        n = len(group)
        # Within this distance bucket, how many are "close enough" isn't meaningful —
        # report count and share of total instead.
        accuracy_by_lev[str(dist)] = {
            "count": n,
            "pct_of_total": round(n / len(results) * 100, 2),
        }

    # Accuracy for "almost correct" (lev ≤ 1, ≤ 2)
    almost = {}
    for threshold in [1, 2, 3]:
        n_close = sum(1 for r in results if r["levenshtein"] <= threshold)
        almost[f"lev_{threshold}_or_less"] = {
            "count": n_close,
            "accuracy": round(n_close / len(results) * 100, 2) if results else 0.0,
        }

    # Count empty/no-answer responses
    empty_count = sum(1 for r in results if not r["predicted"])

    output = {
        "meta": {
            "server": args.server,
            "model": args.model,
            "total_images": len(results),
            "temperature": args.temperature,
        },
        "overall": {
            "correct": total_correct,
            "total": len(results),
            "accuracy_pct": round(overall_accuracy * 100, 2),
            "no_answer": empty_count,
        },
        "almost_correct": almost,
        "accuracy_by_levenshtein_distance": accuracy_by_lev,
        "results": results,
    }

    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"=== Results ===")
    print(f"  Accuracy: {total_correct}/{len(results)} ({overall_accuracy * 100:.2f}%)")
    print(f"  No answer: {empty_count}/{len(results)}")
    for threshold in [1, 2, 3]:
        info = almost[f"lev_{threshold}_or_less"]
        print(f"  Lev ≤ {threshold}: {info['count']}/{len(results)} ({info['accuracy']}%)")
    print(f"  Output: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Benchmark LLM captcha-reading accuracy")
    p.add_argument("--model", default="unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S_think",
                   help="Model name on the llama.cpp server")
    p.add_argument("--server", default=os.environ.get("LLAMA_SERVER_URL", "http://localhost:8033"),
                   help="llama.cpp server base URL (env: LLAMA_SERVER_URL)")
    p.add_argument("--dir", default="categorized",
                   help="Directory with captcha PNGs (filename = ground truth)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of images to test")
    p.add_argument("--sample", type=int, default=None,
                   help="Random sample N images instead of sequential")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for --sample")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Parallel requests to the server")
    p.add_argument("--temperature", type=float, default=None,
                   help="LLM temperature (default: server-defined)")
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="Max completion tokens (thinking models need more)")
    p.add_argument("--enable-thinking", action="store_true", default=None,
                   help="Enable thinking mode via chat_template_kwargs")
    p.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false",
                   help="Disable thinking mode via chat_template_kwargs")
    p.add_argument("--output", default="captcha_results.json",
                   help="Output JSON path")
    p.add_argument("--verbose", action="store_true",
                   help="Print every result line")
    p.add_argument("--debug", action="store_true",
                   help="Print full LLM response details for each request")
    run_bench(p.parse_args())


if __name__ == "__main__":
    main()
