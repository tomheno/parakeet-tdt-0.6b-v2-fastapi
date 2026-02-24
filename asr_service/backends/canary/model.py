import gc

import torch

from .config import DEVICE, MODEL_NAME, MODEL_PRECISION, logger


def _patch_lhotse_sampler():
    """Fix lhotse / PyTorch >=2.10 incompatibility.

    PyTorch 2.10 removed the ``data_source`` parameter from
    ``Sampler.__init__()``, but lhotse's ``CutSampler`` still passes it.
    We monkey-patch the base ``__init__`` to silently drop that argument.
    """
    try:
        from lhotse.dataset.sampling.base import CutSampler

        _original_init = CutSampler.__init__

        def _patched_init(self, *args, **kwargs):
            try:
                return _original_init(self, *args, **kwargs)
            except TypeError:
                # Strip the problematic super().__init__ call by patching Sampler
                import torch.utils.data

                _orig_sampler_init = torch.utils.data.Sampler.__init__

                def _permissive_sampler_init(self_inner, *a, **kw):
                    # Accept (and ignore) data_source / any extra kwargs
                    try:
                        _orig_sampler_init(self_inner)
                    except TypeError:
                        pass

                torch.utils.data.Sampler.__init__ = _permissive_sampler_init
                try:
                    return _original_init(self, *args, **kwargs)
                finally:
                    torch.utils.data.Sampler.__init__ = _orig_sampler_init

        CutSampler.__init__ = _patched_init
        logger.info("Patched lhotse CutSampler for PyTorch >=2.10 compatibility")
    except Exception as exc:
        logger.debug("lhotse patch not needed or failed: %s", exc)


async def _start_canary():
    """Load Canary 1B V2 model, apply optimizations, start batcher.

    Returns (model, batcher) tuple.
    """
    _patch_lhotse_sampler()

    logger.info("Loading %s ...", MODEL_NAME)

    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(model_name=MODEL_NAME, map_location=DEVICE)

    dtype = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32
    model = model.to(dtype=dtype)
    model.eval()
    logger.info("Loaded model with %s weights on %s", MODEL_PRECISION.upper(), DEVICE)

    # Remove CTC timestamps model if not needed (saves ~200MB VRAM)
    import os
    if os.getenv("NO_TIMESTAMPS", "0").lower() in ("1", "true", "yes"):
        if hasattr(model, 'timestamps_asr_model') and model.timestamps_asr_model is not None:
            del model.timestamps_asr_model
            model.timestamps_asr_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Removed CTC timestamps model (NO_TIMESTAMPS=1)")

    # Apply decoder optimizations in two phases:
    # Phase 1 (pre-warmup): greedy + SDPA + KV cache — safe with CUDA graphs
    # Phase 2 (post-warmup): torch.compile decoder — conflicts with graph capture
    from .optimizations import apply_pre_warmup_optimizations, apply_post_warmup_optimizations

    apply_pre_warmup_optimizations(model)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cleanup complete")

    # Start the direct inference batcher (bypasses NeMo DataLoader overhead)
    from .direct_batcher import DirectInferenceBatcher

    batcher = DirectInferenceBatcher(model)
    batcher.warmup()  # pre-compile CUDA graphs for all batch sizes

    # Now safe to compile the decoder (after encoder CUDA graphs are captured)
    apply_post_warmup_optimizations(model)

    batcher.start()
    logger.info(
        "Canary 1B V2 ready on %s — 25 languages, ASR + translation, direct inference",
        next(model.parameters()).device,
    )

    return model, batcher
