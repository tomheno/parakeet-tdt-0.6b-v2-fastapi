"""
Decoder optimizations for Canary 1B V2.

1. TF32 matmul + cuDNN benchmark auto-tuning
2. Switch from beam search (beam_size=1) to pure greedy decoding
3. SDPA with is_causal for self-attn + cross-attn K/V cache + self-attn KV buffer
4. torch.compile decoder FFN (max-autotune, no CUDA graphs)

Toggle via environment variables:
  OPT_TF32=1         — TF32 matmul + cuDNN benchmark (default ON)
  OPT_GREEDY=1       — greedy decoding instead of beam search (default ON)
  OPT_SDPA=1         — SDPA + cross-attn KV cache + causal self-attn (default ON)
  OPT_SELF_KV=1      — self-attn KV buffer: project only new token (default ON)
  OPT_COMPILE_DEC=0        — torch.compile decoder (default OFF)
  COMPILE_DEC_MODE=layers  — compile scope: full|layers|ffn
  COMPILE_DEC_CUDAGRAPHS=1 — enable CUDA graphs in compile (default ON)
  MAX_GEN_DELTA=50         — max output tokens beyond encoder length
"""

import os

import torch
import torch.nn.functional as F

from .config import logger

def _env_bool(name: str, default: bool = True) -> bool:
    val = os.getenv(name, str(int(default)))
    return val.lower() in ("1", "true", "yes")

# Global generation counter for KV cache invalidation
_kv_cache_generation = 0

def clear_kv_cache():
    """Call before each batch to invalidate cross-attention K/V cache."""
    global _kv_cache_generation
    _kv_cache_generation += 1


# ---------------------------------------------------------------------------
# 1. TF32 matmul + cuDNN auto-tune
# ---------------------------------------------------------------------------

def enable_tf32_and_cudnn(model):
    """Enable TF32 tensor cores for FP32 matmul and cuDNN auto-tuning.

    On H100, TF32 gives ~3x faster matmul for any remaining FP32 operations
    (e.g., attention scaling, layer norms, some internal NeMo ops).
    cuDNN benchmark auto-tunes convolution algorithms for FastConformer's
    many Conv1d layers.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    logger.info(
        "Enabled TF32 matmul + cuDNN benchmark (float32_matmul_precision='high')"
    )


# ---------------------------------------------------------------------------
# 2. Greedy decoding (drop beam search overhead)
# ---------------------------------------------------------------------------

def switch_to_greedy(model):
    """Replace beam search (beam_size=1) with pure greedy decoding."""
    from omegaconf import OmegaConf, open_dict

    beam_delta = model.cfg.decoding.beam.get("max_generation_delta", 50)
    # Allow override via env var (lower = faster early stopping)
    max_gen_delta = int(os.getenv("MAX_GEN_DELTA", str(beam_delta)))

    with open_dict(model.cfg.decoding):
        model.cfg.decoding.greedy = OmegaConf.create(
            {
                "max_generation_delta": max_gen_delta,
                "preserve_alignments": False,
                "temperature": None,
                "n_samples": 1,
            }
        )

    model.decoding.change_strategy("greedy")
    logger.info(
        "Decoder: switched to greedy (was beam beam_size=1), "
        "max_generation_delta=%d",
        max_gen_delta,
    )


# ---------------------------------------------------------------------------
# 3. SDPA + cross-attention K/V cache + self-attention KV buffer (combined)
# ---------------------------------------------------------------------------

def _patch_get_memory_states(model):
    """Monkey-patch TransformerDecoder._get_memory_states to avoid double transpose.

    Original: transpose(dim1,2) → cat(dim=2) → transpose(dim1,2)
    Optimized: cat(dim=1) — mathematically identical, avoids 2 transpose ops.
    """
    from nemo.collections.asr.modules.transformer.transformer_decoders import (
        TransformerDecoder,
    )

    def _fast_get_memory_states(self, decoder_states, decoder_mems_list=None, i=0):
        if decoder_mems_list is not None:
            return torch.cat((decoder_mems_list[i], decoder_states), dim=1)
        return decoder_states

    TransformerDecoder._get_memory_states = _fast_get_memory_states
    logger.info("Patched _get_memory_states: eliminated double transpose")


def patch_sdpa_with_kv_cache(model, self_kv_cache=True):
    """Monkey-patch MultiHeadAttention with SDPA and full K/V caching.

    Three optimizations in one patch:
    a) Replace manual matmul attention with F.scaled_dot_product_attention
       (auto-dispatches to FlashAttention-2 on H100 for fp16/bf16)
    b) Cache encoder K/V projections for cross-attention (computed once per batch)
    c) Pre-allocated KV buffer for decoder self-attention: project only new
       token per step instead of re-projecting ALL previous tokens

    Also patches _get_memory_states to eliminate double transpose.

    Args:
        self_kv_cache: Enable pre-allocated self-attention KV buffer (default True).
            Controlled by OPT_SELF_KV env var.
    """
    from nemo.collections.asr.modules.transformer.transformer_modules import (
        MultiHeadAttention,
    )

    # Patch _get_memory_states (double transpose elimination)
    _patch_get_memory_states(model)

    # Add cache attributes to all existing MultiHeadAttention instances
    _original_init = MultiHeadAttention.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self._cached_kv_id = None
        self._cached_k_proj = None
        self._cached_v_proj = None

    MultiHeadAttention.__init__ = _patched_init

    # Tag and patch existing instances (already created during model load)
    # Identify decoder self-attention modules for causal mask optimization.
    # Decoder self-attention is in the transf_decoder, and does NOT have
    # return_xatt_scores=True (that's cross-attention). Encoder self-attention
    # must stay bidirectional (no causal mask).
    _decoder_self_attn_ids = set()
    decoder_wrapper = getattr(model, "transf_decoder", None)
    if decoder_wrapper is not None:
        for name, module in decoder_wrapper.named_modules():
            if isinstance(module, MultiHeadAttention) and not getattr(module, 'return_xatt_scores', False):
                _decoder_self_attn_ids.add(id(module))

    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module._cached_kv_id = None
            module._cached_k_proj = None
            module._cached_v_proj = None
            # Tag decoder self-attention for is_causal optimization
            module._is_decoder_self_attn = id(module) in _decoder_self_attn_ids
            # Self-attention KV buffer attributes
            if self_kv_cache and id(module) in _decoder_self_attn_ids:
                module._self_k_buf = None
                module._self_v_buf = None
                module._self_cache_gen = -1
                module._self_cache_len = 0
                module._self_cache_bs = 0

    logger.info(
        "Tagged %d decoder self-attention modules (self_kv_cache=%s)",
        len(_decoder_self_attn_ids), self_kv_cache,
    )

    # Capture flag in closure
    _self_kv = self_kv_cache

    def _optimized_forward(self, queries, keys, values, attention_mask):
        is_cross_attn = getattr(self, 'return_xatt_scores', False)
        is_decoder_self = getattr(self, '_is_decoder_self_attn', False)
        current_gen = _kv_cache_generation

        # --- Q projection (always computed) ---
        query = self.query_net(queries)

        # --- K/V handling: three paths ---
        kv_transposed = False  # True if K/V already in (B, H, S, D) format

        if is_cross_attn:
            # Cross-attention: cache K/V projections (encoder doesn't change)
            cache_gen = getattr(self, '_cache_gen', -1)
            if cache_gen == current_gen and self._cached_k_proj is not None:
                key = self._cached_k_proj
                value = self._cached_v_proj
            else:
                key = self.key_net(keys)
                value = self.value_net(values)
                self._cached_k_proj = key
                self._cached_v_proj = value
                self._cache_gen = current_gen

        elif is_decoder_self and _self_kv:
            # Decoder self-attention: incremental KV with pre-allocated buffer.
            # Instead of re-projecting ALL memory_states through key_net/value_net
            # every step (O(n²) total), project only the NEW token and write it
            # into a pre-allocated buffer. Use a slice view for SDPA (zero-copy).
            cache_gen = getattr(self, '_self_cache_gen', -1)
            seq_len = keys.shape[1]

            if cache_gen != current_gen:
                # New generation (new batch): project all prompt tokens
                k_proj = self.transpose_for_scores(self.key_net(keys)).contiguous()
                v_proj = self.transpose_for_scores(self.value_net(values)).contiguous()
                bs, nh, sl, hd = k_proj.shape
                # Allocate buffer with room for generation tokens
                buf_len = sl + 200
                # Reuse buffer if shape is compatible, else allocate
                if (self._self_k_buf is None or
                        self._self_k_buf.shape[0] < bs or
                        self._self_k_buf.shape[2] < buf_len):
                    self._self_k_buf = torch.zeros(
                        bs, nh, buf_len, hd,
                        device=k_proj.device, dtype=k_proj.dtype,
                    )
                    self._self_v_buf = torch.zeros(
                        bs, nh, buf_len, hd,
                        device=v_proj.device, dtype=v_proj.dtype,
                    )
                # Write prompt projections into buffer
                self._self_k_buf[:bs, :, :sl, :].copy_(k_proj)
                self._self_v_buf[:bs, :, :sl, :].copy_(v_proj)
                self._self_cache_len = sl
                self._self_cache_gen = current_gen
                self._self_cache_bs = bs
            else:
                # Autoregressive step: project only new token(s)
                cached_len = self._self_cache_len
                bs = self._self_cache_bs
                if seq_len > cached_len:
                    new_tokens = keys[:, cached_len:, :]
                    new_k = self.transpose_for_scores(self.key_net(new_tokens)).contiguous()
                    new_v = self.transpose_for_scores(self.value_net(new_tokens)).contiguous()
                    n = new_k.shape[2]
                    self._self_k_buf[:bs, :, cached_len:cached_len + n, :].copy_(new_k)
                    self._self_v_buf[:bs, :, cached_len:cached_len + n, :].copy_(new_v)
                    self._self_cache_len = cached_len + n

            # Slice view of buffer — zero-copy read for SDPA
            key = self._self_k_buf[:self._self_cache_bs, :, :self._self_cache_len, :]
            value = self._self_v_buf[:self._self_cache_bs, :, :self._self_cache_len, :]
            kv_transposed = True

        else:
            # Encoder self-attention or decoder self-attn without cache
            key = self.key_net(keys)
            value = self.value_net(values)

        # Reshape Q: (batch, seq, hidden) → (batch, num_heads, seq, head_dim)
        query = self.transpose_for_scores(query).contiguous()

        # Reshape K/V if not already in (B, H, S, D) format
        if not kv_transposed:
            key = self.transpose_for_scores(key).contiguous()
            value = self.transpose_for_scores(value).contiguous()

        # Scale: NeMo divides Q,K each by head_dim^0.25 → QK^T / head_dim^0.5
        scale = 1.0 / (self.attn_scale * self.attn_scale)

        # SDPA backend selection for decoder self-attention:
        # - Prefill (q_len == k_len): is_causal=True → cuDNN/Flash backend
        # - Autoregressive (q_len=1, k_len>1): no mask at all (causality is
        #   implicit — future tokens haven't been generated yet)
        # Encoder self-attention & cross-attention: always use additive mask.
        use_causal = False
        attn_mask = None

        if is_decoder_self:
            q_len = query.shape[2]
            k_len = key.shape[2]
            if q_len == k_len and q_len > 1:
                use_causal = True
        elif attention_mask is not None:
            mask = attention_mask.to(query.dtype)
            target_shape = (query.shape[0], query.shape[1], query.shape[2], key.shape[2])
            attn_mask = mask.expand(target_shape)

        context = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            is_causal=use_causal,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=scale,
        )

        # Reshape back: (batch, num_heads, seq, head_dim) → (batch, seq, hidden)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), -1)

        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)

        extra_output = {}
        if self.return_xatt_scores:
            extra_output["xatt_scores"] = None
        return output_states, extra_output

    MultiHeadAttention.forward = _optimized_forward
    logger.info(
        "Patched MultiHeadAttention: SDPA + cross-attn KV cache + "
        "causal self-attn + self-attn KV buffer=%s", self_kv_cache,
    )


# ---------------------------------------------------------------------------
# 3. torch.compile decoder
# ---------------------------------------------------------------------------

def _fix_inplace_residuals(model):
    """Monkey-patch TransformerDecoderBlock.forward_preln to avoid in-place +=.

    NeMo uses `output += residual` which conflicts with CUDA graph tensor
    management. Replace with `output = output + residual` (out-of-place).
    """
    from nemo.collections.asr.modules.transformer.transformer_decoders import (
        TransformerDecoderBlock,
    )

    def _forward_preln_no_inplace(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        residual = decoder_query
        decoder_query = self.layer_norm_1(decoder_query)
        decoder_keys = self.layer_norm_1(decoder_keys)
        self_attn_output, _ = self.first_sub_layer(decoder_query, decoder_keys, decoder_keys, decoder_mask)
        self_attn_output = self_attn_output + residual  # no in-place

        if self.is_adapter_available():
            pack_input = {'x': self_attn_output, 'loc': 'mha', 'att_mask': decoder_mask, 'pos_emb': None}
            pack_input = self.forward_enabled_adapters(pack_input)
            self_attn_output = pack_input['x']

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        enc_dec_attn_output, extra_output = self.second_sub_layer(
            self_attn_output, encoder_states, encoder_states, encoder_mask
        )
        enc_dec_attn_output = enc_dec_attn_output + residual  # no in-place

        residual = enc_dec_attn_output
        enc_dec_attn_output = self.layer_norm_3(enc_dec_attn_output)
        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states = output_states + residual  # no in-place

        if self.is_adapter_available():
            pack_input = {'x': output_states, 'loc': 'post'}
            pack_input = self.forward_enabled_adapters(pack_input)
            output_states = pack_input['x']

        return output_states, extra_output

    TransformerDecoderBlock.forward_preln = _forward_preln_no_inplace
    logger.info("Patched decoder blocks: replaced in-place += with out-of-place + for CUDA graph compatibility")


def compile_decoder(model):
    """torch.compile decoder layers with optional CUDA graphs.

    Compile mode is controlled by COMPILE_DEC_MODE env var:
      "full"    — compile entire decoder forward (max fusion, needs warmup)
      "layers"  — compile each TransformerDecoderBlock (good fusion, modular)
      "ffn"     — compile only FFN sub-layers (safest, least fusion)

    CUDA graph mode controlled by COMPILE_DEC_CUDAGRAPHS env var:
      "1" (default) — max-autotune (includes CUDA graphs)
      "0"           — max-autotune-no-cudagraphs

    Safe to use when COMPILE_ENCODER=0 (no encoder CUDA graphs in the system).
    """
    compile_mode = os.getenv("COMPILE_DEC_MODE", "layers")
    use_cudagraphs = _env_bool("COMPILE_DEC_CUDAGRAPHS", default=True)
    torch_mode = "max-autotune" if use_cudagraphs else "max-autotune-no-cudagraphs"

    # Fix in-place residual ops that break CUDA graphs
    if use_cudagraphs:
        _fix_inplace_residuals(model)

    if hasattr(model, "transf_decoder") and model.transf_decoder is not None:
        wrapper = model.transf_decoder
    elif hasattr(model, "decoder"):
        wrapper = model.decoder
    else:
        logger.warning("No decoder found to compile")
        return

    # TransformerDecoderNM wraps TransformerDecoder in _decoder
    target = getattr(wrapper, "_decoder", wrapper)
    if not hasattr(target, "layers"):
        logger.warning("Decoder has no 'layers' attribute, cannot compile")
        return

    if compile_mode == "full":
        # Compile the entire TransformerDecoder forward
        try:
            compiled = torch.compile(
                target, mode=torch_mode,
                dynamic=True, fullgraph=False,
            )
            # Replace the inner _decoder with compiled version
            if hasattr(wrapper, "_decoder"):
                wrapper._decoder = compiled
            logger.info(
                "Compiled full decoder (%s, cudagraphs=%s)",
                torch_mode, use_cudagraphs,
            )
        except Exception as e:
            logger.warning("Failed to compile full decoder: %s", e)

    elif compile_mode == "layers":
        # Compile each decoder layer (block) individually
        compiled_count = 0
        for i, layer in enumerate(target.layers):
            try:
                compiled_layer = torch.compile(
                    layer, mode=torch_mode,
                    dynamic=True, fullgraph=False,
                )
                target.layers[i] = compiled_layer
                compiled_count += 1
            except Exception as e:
                logger.warning("Failed to compile decoder layer %d: %s", i, e)

        logger.info(
            "Compiled %d/%d decoder layers (%s, cudagraphs=%s)",
            compiled_count, len(target.layers), torch_mode, use_cudagraphs,
        )

    elif compile_mode == "ffn":
        # Compile only FFN sub-layers (most conservative)
        from nemo.collections.asr.modules.transformer.transformer_modules import (
            PositionWiseFF,
        )
        compiled_count = 0
        for i, layer in enumerate(target.layers):
            ffn = getattr(layer, "third_sub_layer", None)
            if ffn is not None and isinstance(ffn, PositionWiseFF):
                try:
                    compiled_ffn = torch.compile(
                        ffn, mode=torch_mode,
                        dynamic=True, fullgraph=False,
                    )
                    layer.third_sub_layer = compiled_ffn
                    compiled_count += 1
                except Exception as e:
                    logger.warning("Failed to compile FFN in layer %d: %s", i, e)

        logger.info(
            "Compiled %d/%d decoder FFN layers (%s, cudagraphs=%s)",
            compiled_count, len(target.layers), torch_mode, use_cudagraphs,
        )
    else:
        logger.warning("Unknown COMPILE_DEC_MODE=%s, skipping", compile_mode)


# ---------------------------------------------------------------------------
# 4. FP8 quantization (H100/H200 only)
# ---------------------------------------------------------------------------

class FP8Linear(torch.nn.Module):
    """Drop-in nn.Linear replacement using FP8 weights + _scaled_mm.

    Weights are stored in float8_e4m3fn (saves 50% memory vs FP16).
    Activations are dynamically quantized to FP8 per-tensor.
    Uses torch._scaled_mm for H100 FP8 tensor cores.
    Falls back to FP16 matmul for shapes _scaled_mm doesn't support.
    """

    def __init__(self, weight_fp8, bias, scale_w):
        super().__init__()
        self.register_buffer("weight_fp8", weight_fp8)  # (out, in) in FP8
        self.register_buffer("scale_w", scale_w)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.out_features = weight_fp8.shape[0]
        self.in_features = weight_fp8.shape[1]

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        # Dynamic per-tensor scale for activations
        with torch.no_grad():
            amax = x_2d.abs().amax()
            # FP8 E4M3 max value is 448.0
            scale_x = (448.0 / amax.clamp(min=1e-12)).float()

        x_fp8 = (x_2d * scale_x).to(torch.float8_e4m3fn)
        inv_scale_x = torch.tensor(1.0, device=x.device, dtype=torch.float32) / scale_x

        try:
            out = torch._scaled_mm(
                x_fp8, self.weight_fp8.t(),
                scale_a=inv_scale_x,
                scale_b=self.scale_w,
                out_dtype=torch.float16,
            )
        except RuntimeError:
            # Fallback for unsupported shapes
            out = torch.mm(x_2d.half(), self.weight_fp8.to(torch.float16).t())

        if self.bias is not None:
            out = out + self.bias

        return out.view(*orig_shape[:-1], self.out_features)


def quantize_to_fp8(model):
    """Replace nn.Linear layers with FP8Linear for H100 FP8 tensor cores.

    Only quantizes layers with large enough dimensions (>= 256) where
    FP8 gives actual speedup. Small layers keep FP16.
    """
    if torch.cuda.get_device_capability()[0] < 9:
        logger.warning("FP8 requires compute capability >= 9.0 (H100/H200)")
        return

    count = 0
    total = 0

    def _replace_linear(module, prefix=""):
        nonlocal count, total
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear):
                total += 1
                # Only quantize large layers where FP8 helps
                if child.in_features >= 256 and child.out_features >= 256:
                    w = child.weight.data  # (out, in) in FP16
                    with torch.no_grad():
                        amax = w.abs().amax()
                        scale_w = (448.0 / amax.clamp(min=1e-12)).float()
                        w_fp8 = (w * scale_w).to(torch.float8_e4m3fn)
                        inv_scale_w = torch.tensor(1.0, device=w.device) / scale_w

                    fp8_layer = FP8Linear(w_fp8, child.bias, inv_scale_w)
                    setattr(module, name, fp8_layer)
                    count += 1
            else:
                _replace_linear(child, full_name)

    _replace_linear(model)
    logger.info("FP8 quantized %d/%d linear layers (>= 256 dims)", count, total)


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def apply_pre_warmup_optimizations(model):
    """Apply optimizations that are safe to run BEFORE encoder CUDA graph warmup.

    greedy + SDPA/KV-cache don't interfere with CUDA graph capture.
    Also raises Dynamo's recompile limit early so encoder warmup benefits too.
    """
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    logger.info("Set torch._dynamo.config.cache_size_limit = 64")

    applied = []

    # TF32 + cuDNN (should be first — affects all subsequent ops)
    if _env_bool("OPT_TF32", default=True):
        enable_tf32_and_cudnn(model)
        applied.append("tf32+cudnn")

    if _env_bool("OPT_GREEDY", default=True):
        switch_to_greedy(model)
        applied.append("greedy")

    if _env_bool("OPT_SDPA", default=True):
        self_kv = _env_bool("OPT_SELF_KV", default=True)
        patch_sdpa_with_kv_cache(model, self_kv_cache=self_kv)
        applied.append("sdpa+xattn_kv" + ("+self_kv" if self_kv else ""))

    if _env_bool("OPT_FP8", default=False):
        quantize_to_fp8(model)
        applied.append("fp8")

    if applied:
        logger.info("Pre-warmup optimizations applied: %s", ", ".join(applied))


def apply_post_warmup_optimizations(model):
    """Apply optimizations that run AFTER encoder warmup.

    When COMPILE_ENCODER=0 (no CUDA graphs), torch.compile on the decoder
    is safe. Uses max-autotune-no-cudagraphs for best kernel selection.
    """
    applied = []

    if _env_bool("OPT_COMPILE_DEC", default=False):
        compile_decoder(model)
        applied.append("compile_decoder")

    if applied:
        logger.info("Post-warmup optimizations applied: %s", ", ".join(applied))


# Keep backward compat
def apply_all_optimizations(model):
    """Apply all optimizations (pre-warmup only). Use the split API for full control."""
    apply_pre_warmup_optimizations(model)
