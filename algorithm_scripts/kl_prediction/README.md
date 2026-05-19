# KL Prediction

Collect conditional KL for Shuffle token pairs and fit hidden-state predictors.

```bash
cd /home/bisharipov/emnlp-experiments
source .venv/bin/activate
python -m algorithm_scripts.kl_prediction.run \
  --samples 2 \
  --output-jsonl outputs/kl_prediction/pairs.jsonl \
  --output-features outputs/kl_prediction/features.pt
```

Train from saved features:

```bash
python -m algorithm_scripts.kl_prediction.run \
  --input-features outputs/kl_prediction/features.pt \
  --train-catboost
```

Defaults: local `Dream-org/Dream-Coder-v0-Instruct-7B`, CUDA if available.

