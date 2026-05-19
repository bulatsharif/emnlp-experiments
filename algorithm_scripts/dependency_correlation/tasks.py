"""Task prompt loaders."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from algorithm_scripts.kl_prediction.shuffle_data import load_shuffle_sample


HUMANEVAL_IDS = (("openai_humaneval", None), ("openai/openai_humaneval", None))
GSM8K_IDS = (("openai/gsm8k", "main"), ("gsm8k", "main"))


@dataclass
class Sample:
    task: str
    messages: list[dict[str, str]]
    target: str
    metadata: dict


def _single_user(task: str, prompt: str, target: str, metadata: dict) -> Sample:
    return Sample(
        task=task,
        messages=[{"role": "user", "content": prompt}],
        target=target,
        metadata=metadata,
    )


def load_shuffle(root: Path, count: int, offset: int, seed: int) -> list[Sample]:
    out = []
    for idx in range(offset, offset + count):
        sample = load_shuffle_sample(root, idx, seed)
        out.append(
            Sample(
                task="shuffle",
                messages=sample.messages,
                target=sample.target,
                metadata={"sample_idx": idx, **sample.metadata},
            )
        )
    return out


def load_humaneval(path: Path | None, count: int, offset: int) -> list[Sample]:
    rows = (
        _read_jsonl(path, count, offset)
        if path
        else _load_hf(HUMANEVAL_IDS, "test", count, offset)
    )
    out = []
    for idx, row in rows:
        prompt = row.get("prompt") or row.get("question") or row.get("text")
        target = row.get("canonical_solution") or row.get("answer") or ""
        if not prompt:
            continue
        out.append(
            _single_user(
                "humaneval",
                "Complete the Python function.\n\n" + prompt,
                target,
                {"sample_idx": idx, "task_id": row.get("task_id")},
            )
        )
    return out


def load_gsm8k(path: Path | None, count: int, offset: int) -> list[Sample]:
    rows = (
        _read_jsonl(path, count, offset)
        if path
        else _load_hf(GSM8K_IDS, "test", count, offset)
    )
    out = []
    for idx, row in rows:
        question = row.get("question") or row.get("prompt") or row.get("text")
        target = row.get("answer") or ""
        if not question:
            continue
        out.append(
            _single_user(
                "gsm8k",
                "Solve the math problem. Give the final answer.\n\n" + question,
                target,
                {"sample_idx": idx},
            )
        )
    return out


def _read_jsonl(path: Path, count: int, offset: int):
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if idx < offset:
                continue
            if idx >= offset + count:
                break
            yield idx, json.loads(line)


def _load_hf(
    candidates: tuple[tuple[str, str | None], ...],
    split: str,
    count: int,
    offset: int,
):
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install datasets or pass a local JSONL file.") from exc

    errors = []
    for name, config in candidates:
        try:
            ds = load_dataset(name, config, split=split)
            break
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    else:
        raise RuntimeError(
            "Could not load benchmark from Hugging Face:\n" + "\n".join(errors)
        )

    end = offset + count
    for idx in range(offset, min(end, len(ds))):
        yield idx, dict(ds[idx])
