"""Weight loading and decode API for Qwen3.5-0.8B bf16 megakernel."""

import json
import os
import struct
from pathlib import Path

import torch

NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048

FA_NUM_Q_HEADS = 8
FA_NUM_KV_HEADS = 2
FA_HEAD_DIM = 256
FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM
FA_QPROJ_SIZE = FA_Q_SIZE * 2
FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM

DN_NUM_HEADS = 16
DN_KEY_DIM = 128
DN_VALUE_DIM = 128
DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM
DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM
DN_CONV_CHANNELS = DN_QK_SIZE * 2 + DN_V_SIZE
DN_CONV_KERNEL = 4

LAYER_TYPE = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

_decode = None
_prefill = None


def _load_op():
    global _decode, _prefill
    if _decode is None or _prefill is None:
        import qwen35_megakernel_bf16_C

        _decode = torch.ops.qwen35_megakernel_bf16_C.decode
        _prefill = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16


def _megakernel_local_only() -> bool:
    return os.getenv("ELFWEAVE_MEGAKERNEL_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}


def _resolve_snapshot_dir(model_name: str, verbose: bool) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    allow_patterns = [
        "config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "model.safetensors-*.safetensors",
    ]
    local_only = _megakernel_local_only()
    try:
        snapshot_dir = snapshot_download(
            repo_id=model_name,
            allow_patterns=allow_patterns,
            local_files_only=local_only,
        )
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            f"Megakernel weights for {model_name} are not cached locally. "
            "Run scripts/setup_megakernel.sh first, or set ELFWEAVE_MEGAKERNEL_LOCAL_ONLY=0 "
            "for a one-time download."
        ) from exc
    if verbose:
        print(f"Using local snapshot: {snapshot_dir}")
    return Path(snapshot_dir)


def _load_state_dict(snapshot_dir: Path):
    from safetensors.torch import load_file

    index_file = snapshot_dir / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
        shard_files = sorted({snapshot_dir / shard for shard in weight_map.values()})
    else:
        shard_files = sorted(snapshot_dir.glob("model.safetensors*"))
    if not shard_files:
        raise FileNotFoundError(f"No model.safetensors files found in {snapshot_dir}")

    state = {}
    for shard_file in shard_files:
        state.update(load_file(str(shard_file), device="cpu"))
    return state


def load_weights(model_name="Qwen/Qwen3.5-0.8B", verbose=True):
    """Load Qwen3.5-0.8B weights as bf16 (no quantization)."""
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoTokenizer

    if verbose:
        print(f"Loading {model_name} weights (bf16)...")
    snapshot_dir = _resolve_snapshot_dir(model_name, verbose=verbose)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir), local_files_only=True)
    state = _load_state_dict(snapshot_dir)

    def _resolve_tensor(name: str):
        candidates = [name]
        if name.startswith("model."):
            candidates.append("model.language_model." + name[len("model."):])
        candidates.append("model.language_model." + name)
        for candidate in candidates:
            if candidate in state:
                return state[candidate]
        raise KeyError(name)

    def _gpu_bf16(name: str):
        return _resolve_tensor(name).to(device="cuda", dtype=torch.bfloat16).contiguous()

    layer_data = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        lt = LAYER_TYPE[i]

        if lt == 1:
            layer_data.append(
                {
                    "type": 1,
                    "ptrs": [
                        _gpu_bf16(p + "input_layernorm.weight"),
                        _gpu_bf16(p + "self_attn.q_proj.weight"),
                        _gpu_bf16(p + "self_attn.k_proj.weight"),
                        _gpu_bf16(p + "self_attn.v_proj.weight"),
                        _gpu_bf16(p + "self_attn.q_norm.weight"),
                        _gpu_bf16(p + "self_attn.k_norm.weight"),
                        _gpu_bf16(p + "self_attn.o_proj.weight"),
                        _gpu_bf16(p + "post_attention_layernorm.weight"),
                        _gpu_bf16(p + "mlp.gate_proj.weight"),
                        _gpu_bf16(p + "mlp.up_proj.weight"),
                        _gpu_bf16(p + "mlp.down_proj.weight"),
                    ],
                }
            )
        else:
            layer_data.append(
                {
                    "type": 0,
                    "ptrs": [
                        _gpu_bf16(p + "input_layernorm.weight"),
                        _gpu_bf16(p + "linear_attn.in_proj_qkv.weight"),
                        _gpu_bf16(p + "linear_attn.in_proj_z.weight"),
                        _gpu_bf16(p + "linear_attn.in_proj_b.weight"),
                        _gpu_bf16(p + "linear_attn.in_proj_a.weight"),
                        _gpu_bf16(p + "linear_attn.conv1d.weight"),
                        _gpu_bf16(p + "linear_attn.A_log"),
                        _gpu_bf16(p + "linear_attn.dt_bias"),
                        _gpu_bf16(p + "linear_attn.norm.weight"),
                        _gpu_bf16(p + "linear_attn.out_proj.weight"),
                        _gpu_bf16(p + "post_attention_layernorm.weight"),
                        _gpu_bf16(p + "mlp.gate_proj.weight"),
                        _gpu_bf16(p + "mlp.up_proj.weight"),
                        _gpu_bf16(p + "mlp.down_proj.weight"),
                    ],
                }
            )

    embed_weight = _gpu_bf16("model.embed_tokens.weight")
    final_norm_weight = _gpu_bf16("model.norm.weight")
    lm_head = None
    for candidate in ("lm_head.weight", "model.language_model.lm_head.weight", "model.lm_head.weight"):
        if candidate in state:
            lm_head = state[candidate]
            break
    if lm_head is None:
        lm_head = embed_weight
    else:
        lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16).contiguous()

    weights = {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head,
        "layer_data": layer_data,
    }

    del state
    torch.cuda.empty_cache()

    if verbose:
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + lm_head.numel()
        print(f"BF16 weights: {total / 1e6:.1f}M params ({total * 2 / 1e6:.0f} MB)")

    return weights, tokenizer


def _pack_layer_weights(layer_data):
    """Pack layer weights into device blob matching LayerWeights struct."""
    ptr_size = 8
    max_ptrs = 14
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class Decoder:
    """Stateful decoder for Qwen3.5-0.8B bf16 megakernel."""

    def __init__(self, weights=None, tokenizer=None, model_name="Qwen/Qwen3.5-0.8B", verbose=True):
        _load_op()

        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, MAX_SEQ_LEN, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        n_dn = sum(1 for t in LAYER_TYPE if t == 0)
        self._dn_states = torch.zeros(n_dn, DN_NUM_HEADS, DN_KEY_DIM, DN_VALUE_DIM, **f32)
        self._conv_bufs = torch.zeros(n_dn, DN_CONV_CHANNELS, DN_CONV_KERNEL, **f32)

        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        max_scratch = max(FA_QPROJ_SIZE, DN_CONV_CHANNELS, HIDDEN_SIZE * 8 + INTERMEDIATE_SIZE)
        self._activations = torch.empty(max_scratch, **f32)
        self._residual = torch.empty(HIDDEN_SIZE, **bf16)
        self._qkv_scratch = torch.empty(max(FA_QPROJ_SIZE, DN_CONV_CHANNELS), **f32)
        self._kv_scratch = torch.empty(FA_KV_SIZE * 2, **f32)
        self._attn_out = torch.empty(max(FA_Q_SIZE, DN_V_SIZE), **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._z_scratch = torch.empty(DN_V_SIZE, **f32)
        self._beta_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._alpha_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._normalized = torch.empty(HIDDEN_SIZE, **f32)

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        self._block_max_vals = torch.empty(1024, **f32)
        self._block_max_idxs = torch.empty(1024, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._out_token = torch.empty(1, **i32)
        self._prefill_buffers = self._allocate_prefill_buffers(MAX_SEQ_LEN)

    def _allocate_prefill_buffers(self, seq_len: int):
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
        return {
            "hidden": torch.empty(seq_len * HIDDEN_SIZE, **bf16),
            "residual": torch.empty(seq_len * HIDDEN_SIZE, **bf16),
            "normalized": torch.empty(seq_len * HIDDEN_SIZE, **bf16),
            "proj_buf": torch.empty(seq_len * mx, **bf16),
            "proj_buf2": torch.empty(seq_len * mx, **bf16),
            "attn_buf": torch.empty(seq_len * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
            "mlp_buf": torch.empty(seq_len * INTERMEDIATE_SIZE, **bf16),
            "dn_out_buf": torch.empty(seq_len * DN_V_SIZE, **bf16),
            "beta_buf": torch.empty(seq_len * DN_NUM_HEADS, **f32),
            "alpha_buf": torch.empty(seq_len * DN_NUM_HEADS, **f32),
            "final_normed": torch.empty(HIDDEN_SIZE, **bf16),
            "hidden_bf16_out": torch.empty(HIDDEN_SIZE, **bf16),
            "lm_bmv": torch.empty(1024, **f32),
            "lm_bmi": torch.empty(1024, **i32),
        }

    def prefill(self, token_ids):
        if not token_ids:
            raise ValueError("prefill requires at least one token")
        if len(token_ids) > MAX_SEQ_LEN:
            raise ValueError(
                f"prompt length {len(token_ids)} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}"
            )
        token_tensor = torch.tensor(token_ids, dtype=torch.int32, device="cuda")
        bufs = self._prefill_buffers
        _prefill(
            self._out_token,
            token_tensor,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._fa_k_cache,
            self._fa_v_cache,
            self._dn_states,
            self._conv_bufs,
            bufs["hidden"],
            bufs["residual"],
            bufs["normalized"],
            bufs["proj_buf"],
            bufs["proj_buf2"],
            bufs["attn_buf"],
            bufs["mlp_buf"],
            bufs["dn_out_buf"],
            bufs["beta_buf"],
            bufs["alpha_buf"],
            bufs["final_normed"],
            bufs["hidden_bf16_out"],
            bufs["lm_bmv"],
            bufs["lm_bmi"],
        )
        self._hidden.copy_(bufs["hidden_bf16_out"])
        self._position = len(token_ids)
        return self._out_token.item()

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        if self._position >= MAX_SEQ_LEN:
            raise ValueError(
                f"decode position {self._position} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}"
            )
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._fa_k_cache,
            self._fa_v_cache,
            self._dn_states,
            self._conv_bufs,
            self._hidden,
            self._activations,
            self._residual,
            self._qkv_scratch,
            self._kv_scratch,
            self._attn_out,
            self._mlp_inter,
            self._z_scratch,
            self._beta_scratch,
            self._alpha_scratch,
            self._normalized,
            self._barrier_counter,
            self._barrier_generation,
            self._block_max_vals,
            self._block_max_idxs,
            self._lm_sync_counter,
            self._position,
            MAX_SEQ_LEN,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not ids:
            return ""
        if len(ids) > MAX_SEQ_LEN:
            raise ValueError(
                f"prompt length {len(ids)} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}"
            )
        max_steps = min(max_tokens, max(0, MAX_SEQ_LEN - len(ids) + 1))
        if max_steps <= 0:
            return ""

        out = []
        next_id = self.prefill(ids)
        eos = self.tokenizer.eos_token_id
        if next_id != eos:
            out.append(next_id)

        for _ in range(max_steps - 1):
            next_id = self.step(next_id)
            if next_id == eos:
                break
            out.append(next_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)
