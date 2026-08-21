# quant rule: fp8-rowwise@1  — THE ENTIRE DOCUMENT

eligible:  2-D weight tensors of the DENOISER component
           (all other components pass through unchanged)

transform, per eligible tensor W: BF16 [out, in]:
    emit  W             as  F8_E4M3  [out, in]     # same name, same shape
    emit  W + "_scale"  as  F32      [out]         # new per-row scale twin

everything else: unchanged.

inverse (this line is what makes "derivable via" computable):
    W_bf16[r, c] = W_f8[r, c] * W_scale[r]

Note: no SDXL key, no qwen key, no model name anywhere above.
Apply it to ANY topology and it computes that architecture's expected fp8 header.
