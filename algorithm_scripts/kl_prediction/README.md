# KL Prediction

Collect conditional KL for token pairs and fit hidden-state predictors.

Algorithm:

1. Load samples from Shuffle, HumanEval, or GSM8K.
2. Append masked generation tokens to the prompt.
3. Run Dream and get logits plus final hidden states.
4. Pick the highest-confidence masked tokens for the current step.
5. For selected token pairs, unmask one token and measure KL shift at the other.
6. Save pair records and `[anchor_hidden, target_hidden]` features.
7. Fit one linear ridge probe per task by default.
8. Report train/test metrics separately for each task.
9. Optionally fit aggregate CatBoost with `--train-catboost`.

```bash
cd /home/bisharipov/emnlp-experiments
source .venv/bin/activate
python -m algorithm_scripts.kl_prediction.run \
  --tasks shuffle \
  --samples 2 \
  --output-jsonl outputs/kl_prediction/pairs.jsonl \
  --output-features outputs/kl_prediction/features.pt
```

All benchmarks:

```bash
python -m algorithm_scripts.kl_prediction.run \
  --tasks shuffle,humaneval,gsm8k \
  --samples 10 \
  --output-jsonl outputs/kl_prediction/all_pairs.jsonl \
  --output-features outputs/kl_prediction/all_features.pt
```

Train from saved features:

```bash
python -m algorithm_scripts.kl_prediction.run \
  --input-features outputs/kl_prediction/features.pt \
  --train-catboost
```

The linear report is grouped by benchmark task.

Auto-downloads: `openai_humaneval`, `openai/gsm8k`.

Defaults: local `Dream-org/Dream-Coder-v0-Instruct-7B`, CUDA if available.
