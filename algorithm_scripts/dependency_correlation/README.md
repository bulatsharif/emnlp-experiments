# Dependency Correlation

Symmetric KL dependency vs cosine similarity.

Algorithm:

1. Load samples from Shuffle, HumanEval, GSM8K, MATH500, MT-Bench, or IFEval.
2. Append masked generation tokens to each prompt.
3. Run the selected diffusion model and get logits plus final hidden states.
4. By default, pick the highest-confidence masked tokens for the current step.
5. For each selected pair `(i, j)`, unmask `j` and measure KL at `i`.
6. Unmask `i` and measure KL at `j`.
7. Average both KL values into symmetric true dependency.
8. Compute hidden-state cosine and logit cosine for the pair.
9. Report Spearman and Pearson correlations with true dependency.

You can switch off confidence-based selection with `--pair-selection any`, which uses the first available masked positions and the first available combinations instead of sorting by confidence.
You can also disable the iterative unmask-and-reforward loop with `--single-forward-only`, which collects pairs from just the first masked forward pass while still running the conditioned forwards needed for KL dependency.

```bash
cd /workspace/tasks/emnlp/emnlp-experiments
source /workspace/tasks/emnlp/.venv/bin/activate
python -m algorithm_scripts.dependency_correlation.run \
  --tasks shuffle \
  --samples 10 \
  --output-jsonl outputs/dependency_correlation/pairs.jsonl \
  --output-csv outputs/dependency_correlation/summary.csv
```

All benchmarks:

```bash
python -m algorithm_scripts.dependency_correlation.run \
  --tasks shuffle,humaneval,gsm8k,math500,mtbench \
  --samples 10 \
  --output-csv outputs/dependency_correlation/summary.csv
```

Auto-downloads: `openai_humaneval`, `openai/gsm8k`, `HuggingFaceH4/MATH-500`, `HuggingFaceH4/mt_bench_prompts`, `google/IFEval`.

Optional local overrides:

```bash
python -m algorithm_scripts.dependency_correlation.run \
  --tasks humaneval,gsm8k,math500,mtbench \
  --humaneval-jsonl data/humaneval.jsonl \
  --gsm8k-jsonl data/gsm8k.jsonl \
  --math500-jsonl data/math500.jsonl \
  --mtbench-jsonl data/mtbench.jsonl \
  --ifeval-jsonl data/ifeval.jsonl
```

For this experiment, `mtbench` is loaded as standalone prompt turns rather than a judged two-turn chat session.
That gives us a compact, popular, open-ended chat-style prompt set without inventing assistant history.

Runner for the local workspace models:

```bash
bash algorithm_scripts/dependency_correlation/run_requested_models.sh --samples 10
```

4-GPU parallel runner with top-level job progress plus per-job sample progress in logs:

```bash
bash algorithm_scripts/dependency_correlation/run_requested_models.sh \
  --samples 10 \
  --single-forward-only \
  --pair-selection any \
  --gpus 0,1,2,3 \
  --max-parallel 4
```

Notes:

1. The bash runner parallelizes at the `(model, task)` level.
   With the default benchmark set in this workspace, it schedules 12 independent jobs across 4 tasks and 3 models.
2. Each job is pinned to one GPU via `CUDA_VISIBLE_DEVICES`, which is a clean fit for 7B/8B diffusion LMs on 80 GB H100s.
3. The terminal shows a top-level jobs progress bar, while each background job writes its own `tqdm` sample progress bar into `outputs/dependency_correlation/<timestamp>/logs/*.log`.
4. Output layout is:
   `outputs/dependency_correlation/<timestamp>/<model>/<task>/pairs.jsonl`
   `outputs/dependency_correlation/<timestamp>/<model>/<task>/summary.csv`
   `outputs/dependency_correlation/<timestamp>/<model>/summary.csv`
   `outputs/dependency_correlation/<timestamp>/combined_summary.csv`
   `outputs/dependency_correlation/<timestamp>/paper_table.csv`
   `combined_summary.csv` keeps the full statistics, including p-values.
   `paper_table.csv` is the concise manuscript-facing table without p-values.
