# captcha-bench

Benchmark LLM accuracy at reading CAPTCHA images via a llama.cpp server.

Images are stored in `categorized/*.png` where the filename (minus `.png`) is the ground-truth text. Comparison is **case-insensitive**.

## Setup

```bash
uv venv
. .venv/bin/activate
uv pip install requests
cp .env.example .env  # edit with your server URL
```

Create a `.env` file with:

```
LLAMA_SERVER_URL=http://your-server:port
```

## Usage

```bash
# Quick test with 10 random captchas
python3 captcha_bench.py --sample 10

# Full run on a specific model
python3 captcha_bench.py --model MODEL_NAME --output results.json

# Disable thinking (faster, ~3x)
python3 captcha_bench.py --no-enable-thinking --sample 50

# Debug mode — print full LLM response per request
python3 captcha_bench.py --sample 5 --debug --verbose
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S_think` | Model name on the llama.cpp server |
| `--server` | env `LLAMA_SERVER_URL` or `http://localhost:8033` | llama.cpp server base URL |
| `--dir` | `categorized` | Directory with captcha PNGs |
| `--limit N` | all | Limit number of images |
| `--sample N` | — | Random sample N images |
| `--seed` | `42` | Random seed for `--sample` |
| `--concurrency` | `4` | Parallel requests |
| `--max-tokens` | `16384` | Max completion tokens (thinking models need more) |
| `--temperature` | server default | LLM temperature (omit to use server setting) |
| `--enable-thinking` | — | Enable thinking via `chat_template_kwargs` |
| `--no-enable-thinking` | — | Disable thinking via `chat_template_kwargs` |
| `--output` | `captcha_results.json` | Output JSON path |
| `--verbose` | off | Print every result line |
| `--debug` | off | Print full LLM response per request |

## Output

JSON with:

- **`overall`** — correct count, total, accuracy %, no-answer count
- **`almost_correct`** — cumulative accuracy for Levenshtein ≤ 1, 2, 3
- **`accuracy_by_levenshtein_distance`** — count and % of total per distance bucket
- **`results`** — per-image: file, ground_truth, predicted, levenshtein, correct, error

## Multi-model comparison

```bash
python3 captcha_bench.py --model MODEL_A --output results_a.json --sample 100
python3 captcha_bench.py --model MODEL_B --output results_b.json --sample 100
```

Each JSON is self-contained with model metadata, making diffs straightforward.
