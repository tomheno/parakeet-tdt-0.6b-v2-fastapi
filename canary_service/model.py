import gc
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app):
    """Load Canary 1B V2 on startup, release GPU on shutdown."""
    _patch_lhotse_sampler()

    logger.info("Loading %s ...", MODEL_NAME)

    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(model_name=MODEL_NAME, map_location=DEVICE)

    dtype = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32
    model = model.to(dtype=dtype)
    model.eval()
    logger.info("Loaded model with %s weights on %s", MODEL_PRECISION.upper(), DEVICE)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cleanup complete")

    app.state.asr_model = model

    # Start the direct inference batcher (bypasses NeMo DataLoader overhead)
    from .direct_batcher import DirectInferenceBatcher

    batcher = DirectInferenceBatcher(model)
    batcher.warmup()  # pre-compile CUDA graphs for all batch sizes
    batcher.start()
    app.state.batcher = batcher
    logger.info(
        "Canary 1B V2 ready on %s — 25 languages, ASR + translation, direct inference",
        next(model.parameters()).device,
    )

    try:
        yield
    finally:
        logger.info("Releasing GPU memory and shutting down")
        batcher.stop()
        del app.state.batcher
        del app.state.asr_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
