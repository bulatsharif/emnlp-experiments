# Dependency Correlation

Symmetric KL dependency vs cosine similarity.

Algorithm:

1. Load samples from Shuffle, HumanEval, or GSM8K.
2. Append masked generation tokens to each prompt.
3. Run Dream and get logits plus final hidden states.
4. Pick the highest-confidence masked tokens for the current step.
5. For each selected pair `(i, j)`, unmask `j` and measure KL at `i`.
6. Unmask `i` and measure KL at `j`.
7. Average both KL values into symmetric true dependency.
8. Compute hidden-state cosine and logit cosine for the pair.
9. Report Spearman and Pearson correlations with true dependency.

```bash
cd /home/bisharipov/emnlp-experiments
source .venv/bin/activate
python -m algorithm_scripts.dependency_correlation.run \
  --tasks shuffle \
  --samples 10 \
  --output-jsonl outputs/dependency_correlation/pairs.jsonl \
  --output-csv outputs/dependency_correlation/summary.csv
```

All benchmarks:

```bash
python -m algorithm_scripts.dependency_correlation.run \
  --tasks shuffle,humaneval,gsm8k \
  --samples 10 \
  --output-csv outputs/dependency_correlation/summary.csv
```

Auto-downloads: `openai_humaneval`, `openai/gsm8k`.

Optional local overrides:

```bash
python -m algorithm_scripts.dependency_correlation.run \
  --tasks humaneval,gsm8k \
  --humaneval-jsonl data/humaneval.jsonl \
  --gsm8k-jsonl data/gsm8k.jsonl
```
