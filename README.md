# Draw Things M5 SDXL LoRA training: MFA / Neural Accelerator fix + latent cache

Patches against `drawthingsai/draw-things-community` at commit `4e98672`.

## What this is
On Apple M5, SDXL LoRA training with Metal FlashAttention (MFA) and Neural Accelerator (NAX) dispatch enabled produced non-finite gradients in the MFA backward path. GradScaler then skipped optimizer steps and the LoRA stayed effectively unchanged.

These patches:
1. Keep MFA kernels but disable Neural Accelerator dispatch for SD/SDXL training by default (`DRAWLORA_ENABLE_MFA=1` to opt back in).
2. Add diagnostics: per-parameter NaN dump, attention-level switch (`DRAWLORA_ATTN_LEVEL=quantized|scale1|scaleMerged|none`), skip-fail-fast controls.
3. Add a CLI `export lora` subcommand (f32 checkpoint to safetensors).
4. Cache VAE latents across runs, keyed by dataset contents and encoding settings (`~/Library/Caches/DrawThingsLoRA/`, opt out with `DRAWLORA_NO_DATASET_CACHE=1`).

## Verified configuration
- Apple M5, macOS 27.0.1
- SDXL-family 8-bit base model, fp16 LoRA, rank 32, 1024 buckets, aspect-ratio training
- `DRAWLORA_ENABLE_MFA=1`, `DRAWLORA_ATTN_LEVEL=quantized`, clip-skip 2

Results: 900 steps, 0 skipped optimizer steps, ~3.5-3.7 s/step steady state.
`scaleMerged` / `scale1` attention levels reproduced 8 skipped steps in a 32-step benchmark; `quantized` was stable.

## Apply
```bash
git checkout -b m5-mfa-nax-fix 4e98672
git am 0001-*.patch 0002-*.patch 0003-*.patch 0004-*.patch
```

## Caveats
- This is a workaround plus trainer fixes, not a kernel-level fix of MFA backward.
- Only validated for SDXL LoRA training on one M5 machine. Other model families and longer runs are untested.
- Diagnostic prints are intentionally verbose; they are gated by environment variables where practical.

## Related
Upstream discussion with the MFA maintainers: https://github.com/liuliu/ccv/issues/287
