"""Minimal local loader for ParallelBench waiting_line/shuffle samples."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ProbeSample:
    messages: list[dict[str, str]]
    target: str
    metadata: dict


def _list_text(items: list[str]) -> str:
    return "[" + ", ".join(json.dumps(str(item)) for item in items) + "]"


def _load_words(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    if isinstance(data[0], list):
        return [" ".join(parts) for parts in itertools.product(*data)]
    return data


def _task_config(repo_root: Path) -> dict:
    path = (
        repo_root
        / "ParallelBench/parallelbench/datasets/data/task_configs/test/waiting_line.yaml"
    )
    raw = yaml.safe_load(path.read_text())
    global_cfg = raw["global_config"]
    task_cfg = {**global_cfg, **raw["tasks"]["shuffle"]}
    words_path = (
        repo_root
        / "ParallelBench/parallelbench/datasets/data/resources"
        / task_cfg["words"]
    )
    task_cfg["words"] = _load_words(words_path)
    return task_cfg


def _make_shuffle_sample(rng: random.Random, cfg: dict) -> dict:
    length = rng.randint(cfg["min_length"], cfg["max_length"])
    base = rng.sample(cfg["words"], length)
    shuffled = base[:]
    while shuffled == base and len(set(base)) > 1:
        rng.shuffle(shuffled)
    return {
        "input": {"context": _list_text(base)},
        "target": _list_text(shuffled),
        "metadata": {"length": length},
    }


def load_shuffle_sample(repo_root: Path, sample_idx: int, seed: int) -> ProbeSample:
    cfg = _task_config(repo_root)
    rng = random.Random(seed)
    sample = None
    for _ in range(sample_idx + 1):
        sample = _make_shuffle_sample(rng, cfg)
    assert sample is not None

    icl = cfg["icl_example"]
    prompt = cfg["prompt"].format(**sample["input"]).replace("\\n", "\n")
    icl_answer = icl["answer"]["example"]
    messages = [
        {
            "role": "user",
            "content": cfg["prompt"].format(**icl["input"]).replace("\\n", "\n"),
        },
        {"role": "assistant", "content": icl_answer},
        {"role": "user", "content": prompt},
    ]
    return ProbeSample(messages=messages, target=sample["target"], metadata=sample["metadata"])
