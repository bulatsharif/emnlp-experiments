# Dependency Correlation Pipeline

This note documents the exact experiment design for the command:

```bash
bash emnlp-experiments/algorithm_scripts/dependency_correlation/run_requested_models.sh \
  --samples 100 \
  --gpus 0,1,2,3 \
  --max-parallel 4 \
  --single-forward-only
```

Because `--pair-selection` is omitted, the run uses the default:

```text
pair_selection = confidence
```

## 1. Resolved Configuration

The bash runner resolves the following effective setup:

- Models:
  - `Dream-Coder-v0-Instruct-7B`
  - `LLaDA-8B-Instruct`
  - `LLaDA-MoE-7B-A1B-Instruct`
- Benchmarks:
  - `gsm8k`
  - `humaneval`
  - `math500`
  - `mtbench`
- Samples per benchmark: `100`
- GPUs: `0,1,2,3`
- Maximum concurrent jobs: `4`
- `max_new_tokens = 32`
- `generation_steps = 8`
- `single_forward_only = true`
- `tokens_per_step = 5`
- `max_pairs_per_step = 8`
- `pair_selection = confidence`
- `condition_batch_size = 8`
- `device = cuda`
- `dtype = auto`, which resolves to `bfloat16` on CUDA
- `sdpa_backend = auto`
- `shift_logits = auto`, which resolves to `True` for these Dream/LLaDA models

## 2. Outer Scheduling

The runner constructs one job per `(model, benchmark)` pair.

- Number of models: `3`
- Number of benchmarks: `4`
- Total jobs: `12`

Jobs are:

1. Dream-Coder-v0-Instruct-7B x GSM8K
2. Dream-Coder-v0-Instruct-7B x HumanEval
3. Dream-Coder-v0-Instruct-7B x MATH500
4. Dream-Coder-v0-Instruct-7B x MT-Bench
5. LLaDA-8B-Instruct x GSM8K
6. LLaDA-8B-Instruct x HumanEval
7. LLaDA-8B-Instruct x MATH500
8. LLaDA-8B-Instruct x MT-Bench
9. LLaDA-MoE-7B-A1B-Instruct x GSM8K
10. LLaDA-MoE-7B-A1B-Instruct x HumanEval
11. LLaDA-MoE-7B-A1B-Instruct x MATH500
12. LLaDA-MoE-7B-A1B-Instruct x MT-Bench

With `--max-parallel 4` and `4` visible GPUs, at most `4` jobs run simultaneously.
Each job is pinned to exactly one GPU with `CUDA_VISIBLE_DEVICES=<gpu_id>`.

## 3. Per-Job Data Loading

Each job runs:

```bash
python -m algorithm_scripts.dependency_correlation.run \
  --model <one local model path> \
  --tasks <one benchmark> \
  --samples 100 \
  --max-new-tokens 32 \
  --generation-steps 8 \
  --single-forward-only \
  --tokens-per-step 5 \
  --max-pairs-per-step 8 \
  --pair-selection confidence \
  --condition-batch-size 8 \
  --device cuda \
  --dtype auto \
  --sdpa-backend auto
```

Benchmark loading behavior:

- `gsm8k`: first `100` examples from the Hugging Face `test` split.
- `humaneval`: first `100` examples from the Hugging Face `test` split.
- `math500`: first `100` examples from the Hugging Face `test` split.
- `mtbench`: first `100` flattened prompt turns from `HuggingFaceH4/mt_bench_prompts`.

Prompt formatting:

- `gsm8k`: `"Solve the math problem. Give the final answer.\n\n" + question`
- `humaneval`: `"Complete the Python function.\n\n" + prompt`
- `math500`: `"Solve the competition math problem. Give the final answer.\n\n" + problem`
- `mtbench`: raw prompt turn as a standalone user prompt

## 4. Per-Sample Input Construction

For each sample:

1. The prompt is tokenized.
2. `32` mask tokens are appended to the prompt.
3. The model input becomes:

```text
[prompt tokens] + [32 masked generation slots]
```

The experiment records the start position of the masked suffix.

## 5. Single-Forward-Only Consequence

Although `generation_steps = 8` is passed, `--single-forward-only` changes the actual behavior:

- only the first masked forward pass is used
- the iterative unmask-and-reforward generation loop is disabled
- no second generation step is entered

So, algorithmically, each prompt uses exactly one base masked snapshot.

## 6. Base Forward Pass

For that one base forward pass, the code extracts:

- logits at all masked positions
- final hidden states at all masked positions
- attention maps, when supported by the model wrapper

For each masked position `k`, it computes:

- the top predicted token `t_k`
- the confidence `c_k = max softmax probability at position k`

Before computing log-probabilities, the mask-token logit itself is removed from consideration.

## 7. Confidence-Based Position Selection

Because `pair_selection = confidence`, masked positions are chosen by confidence.

There are `32` masked slots total.

1. Sort the `32` masked positions by decreasing confidence.
2. Keep the top `tokens_per_step = 5` masked positions.

Call these selected masked positions:

```text
p1, p2, p3, p4, p5
```

## 8. Confidence-Based Pair Selection

All unordered pairs among these 5 positions are formed:

```text
C(5, 2) = 10 pairs
```

For each pair `(i, j)`, assign the pair score:

```text
pair_score(i, j) = c_i + c_j
```

Then:

1. sort the 10 pairs by decreasing `pair_score`
2. keep the top `max_pairs_per_step = 8` pairs

So each sample contributes at most `8` selected pairs.

## 9. Proxy Metrics From the Base Forward

For each selected pair `(i, j)`, three proxy metrics are computed from the base forward:

1. `attention_score`
   - mean attention between positions `i` and `j`
   - averaged over layers and heads
   - symmetrized as `0.5 * (A_ij + A_ji)`

2. `hidden_cosine`
   - absolute cosine similarity between final hidden states at `i` and `j`

3. `logit_cosine`
   - absolute cosine similarity between the masked-position logits at `i` and `j`
   - with the mask-token logit zeroed out first

## 10. True Dependency Target

The experiment does not stop at proxy metrics.
It also computes a target dependency value using conditioned forwards.

For each selected pair `(i, j)`:

1. create one conditioned copy where position `j` is replaced by its top predicted token `t_j`
2. measure the KL effect on the distribution at `i`
3. create another conditioned copy where position `i` is replaced by its top predicted token `t_i`
4. measure the KL effect on the distribution at `j`

This yields:

- `KL_i_given_j`
- `KL_j_given_i`

The symmetric dependency target is:

```text
true_dependency(i, j) = 0.5 * [KL_i_given_j + KL_j_given_i]
```

## 11. Conditioned Forward Batching

Each pair requires two conditioned sequences.

Since:

- selected pairs per sample = `8`
- conditioned sequences per pair = `2`

each sample produces:

```text
2 x 8 = 16 conditioned sequences
```

The code batches pairs in chunks of `condition_batch_size = 8` pairs.
Therefore, in this setup, each sample uses:

- `1` base forward call
- `1` conditioned forward call containing `16` conditioned sequences

So the experiment is not "one total model call per prompt".
It is:

- one base masked forward snapshot
- plus one batched conditioned forward for dependency estimation

## 12. Per-Sample Record Count

Because this setup is:

- single-forward-only
- 5 selected positions
- top 8 pairs retained

each sample contributes exactly:

```text
8 pair records
```

assuming the sample runs successfully.

With `100` samples per `(model, benchmark)` job:

```text
100 x 8 = 800 pair records per job
```

This matches the `pairs = 800` values seen in the paper table.

## 13. Correlation Computation

After all pair records for one job are collected, the job aggregates them by benchmark and computes:

- Spearman correlation between `true_dependency` and `attention_score`
- Spearman correlation between `true_dependency` and `logit_cosine`
- Spearman correlation between `true_dependency` and `hidden_cosine`
- Pearson correlation for the same three comparisons
- corresponding p-values

Non-finite values are filtered before correlation.

## 14. Output Files

Each job writes:

- `<outdir>/<model>/<task>/pairs.jsonl`
- `<outdir>/<model>/<task>/summary.csv`

After all 12 jobs finish successfully, the bash runner merges everything into:

- `<outdir>/<model>/summary.csv`
- `<outdir>/combined_summary.csv`
- `<outdir>/paper_table.csv`

## 15. Total Experiment Scale

For this exact command, the intended total scale is:

- prompts per benchmark per model: `100`
- pair records per benchmark per model: `800`
- jobs: `12`
- total prompts overall:

```text
3 models x 4 benchmarks x 100 = 1200 prompts
```

- total pair records overall:

```text
3 models x 4 benchmarks x 800 = 9600 pair records
```

## 16. Interpretation of This Setup

This setup measures dependency structure in a restricted one-step regime.

It answers the question:

```text
Given one masked snapshot of the prompt plus 32 future masked slots,
if we focus on the 5 most confident masked positions and the 8 strongest
confidence-based pairs among them, how well do attention similarity,
logit similarity, and hidden-state similarity predict the true symmetric
dependency measured by conditional KL?
```

Relative to the full iterative version, this design:

- removes later-step generation dynamics
- keeps dependency measurement itself intact
- focuses on early, high-confidence masked positions
- yields a fixed and interpretable `800` records per job
