"""Compatibility helpers for probing local Dream checkpoints."""

from __future__ import annotations

import os
import site
import sys
from contextlib import nullcontext
from pathlib import Path

import torch


FORCE_MATH_SDPA = False


def resolve_shift_logits(mode: str, model_name: str) -> bool:
    if mode == "yes":
        return True
    if mode == "no":
        return False
    return "dream" in model_name.lower() or "llada" in model_name.lower()


def _configure_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/huggingface/modules")


def _add_project_venv_to_path() -> None:
    _configure_hf_cache()
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
        root / ".venv" / "Lib" / "site-packages",
    ]
    for path in candidates:
        if path.exists():
            site.addsitedir(str(path))
            if str(path) in sys.path:
                sys.path.remove(str(path))
            sys.path.insert(0, str(path))
            return


def _patch_rope_default() -> None:
    """Dream remote code may expect the old Transformers RoPE key."""
    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return

    if "default" in rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def default_rope(config, device, seq_len=None, **kwargs):
        del seq_len, kwargs
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        base = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        return inv_freq, 1.0

    rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_rope


def _patch_generation_update() -> None:
    """DreamGenerationConfig.validate ignores new HF kwargs."""
    try:
        from transformers.generation.configuration_utils import GenerationConfig
    except Exception:
        return

    if getattr(GenerationConfig.update, "_dream_compat", False):
        return

    def update_compat(self, defaults_only=False, allow_custom_entries=False, **kwargs):
        used = []
        for key, value in kwargs.items():
            if allow_custom_entries and not hasattr(self, key):
                setattr(self, key, value)
                used.append(key)
            elif hasattr(self, key) and (not defaults_only or getattr(self, key) is None):
                setattr(self, key, value)
                used.append(key)

        try:
            self.validate(user_set_attributes=set(used))
        except TypeError as exc:
            if "user_set_attributes" not in str(exc):
                raise
            self.validate()

        return {key: value for key, value in kwargs.items() if key not in used}

    update_compat._dream_compat = True
    GenerationConfig.update = update_compat


def _configure_attention(device: str, backend: str) -> None:
    global FORCE_MATH_SDPA
    FORCE_MATH_SDPA = backend == "math"
    if not str(device).startswith("cuda") or backend != "math":
        return
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def load_dream_model(args):
    _configure_hf_cache()
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        _add_project_venv_to_path()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError:
            raise RuntimeError("transformers is not importable") from exc

    _patch_rope_default()
    _patch_generation_update()

    model_path = Path(args.model)
    model_id = str(model_path if model_path.exists() else args.model)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if args.dtype == "auto" and device.startswith("cuda") else (
        torch.float32 if args.dtype == "auto" else getattr(torch, args.dtype)
    )
    _configure_attention(device, args.sdpa_backend)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=not args.allow_downloads
    )
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=not args.allow_downloads,
    ).to(device)
    model.eval()

    mask_id = args.mask_token_id
    if mask_id is None:
        mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("Could not infer mask token id; pass --mask-token-id.")

    return model, tokenizer, int(mask_id), device


def forward_logits_hidden(model, input_ids, shift_logits: bool):
    position_ids = torch.arange(
        input_ids.shape[-1], dtype=torch.long, device=input_ids.device
    ).unsqueeze(0).expand(input_ids.shape[0], -1)

    context = nullcontext()
    if input_ids.device.type == "cuda" and FORCE_MATH_SDPA:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            context = sdpa_kernel(SDPBackend.MATH)
        except Exception:
            context = nullcontext()

    with context:
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    logits = output.logits
    if shift_logits:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    hidden = output.hidden_states
    if isinstance(hidden, (tuple, list)):
        hidden = hidden[-1]
    return logits, hidden
