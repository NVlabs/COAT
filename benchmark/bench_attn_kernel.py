import argparse
import math
import warnings
    
import torch
import triton
import triton.language as tl
from collections import defaultdict

try:
    from flash_attn.flash_attn_interface import flash_attn_func as flash2
    from flash_attn_interface import flash_attn_func as flash3
except:
    warnings.warn("flash_attn is not installed")

try:
    import flashinfer
except:
    warnings.warn("flashinfer is not installed")

from coat.attention.triton_attn import attention_forward
from coat.attention.triton_attn_tma import tma_attention_forward

# Supported methods
BENCH_CONFIGS = {
    "flashattn2-bf16": {
        "line_vals": "flashattn2-bf16",
        "line_names": "FlashAttn2-BF16",
        "color": ("red", "-"),
    },
    "flashattn3-bf16": {
        "line_vals": "flashattn3-bf16",
        "line_names": "FlashAttn3-BF16",
        "color": ("blue", "-"),
    },
    "flashattn3-fp8": {
        "line_vals": "flashattn3-fp8",
        "line_names": "FlashAttn3-FP8",
        "color": ("purple", "-"),
    },
    "triton-bf16": {
        "line_vals": "triton-bf16",
        "line_names": "Triton-BF16",
        "color": ("green", "-"),
    },
    "triton-bf16-tma": {
        "line_vals": "triton-bf16-tma",
        "line_names": "Triton-BF16-TMA",
        "color": ("black", "-"),
    },
    "triton-fp8": {
        "line_vals": "triton-fp8",
        "line_names": "Triton-FP8",
        "color": ("orange", "-"),
    },
    "triton-fp8-tma": {
        "line_vals": "triton-fp8-tma",
        "line_names": "Triton-FP8-TMA",
        "color": ("pink", "-"),
    },
    "flashinfer2-bf16": {
        "line_vals": "flashinfer2-bf16",
        "line_names": "FlashInfer2-BF16",
        "color": ("brown", "-"),
    },
    "flashinfer3-bf16": {
        "line_vals": "flashinfer3-bf16",
        "line_names": "FlashInfer3-BF16",
        "color": ("brown", "-"),
    },
    "flashinfer2-fp8": {
        "line_vals": "flashinfer2-fp8",
        "line_names": "FlashInfer2-FP8",
        "color": ("brown", "-"),
    },
    "flashinfer3-fp8": {
        "line_vals": "flashinfer3-fp8",
        "line_names": "FlashInfer3-FP8",
        "color": ("brown", "-"),
    },
}

# FLOP Estimation
def compute_flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False):
    avg_seqlen = (seqlen_k if not causal else (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2)
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)

def benchmarker(seqlen, batch_size, model_dim, dropout_p, headdim, provider, gen_dtype=torch.bfloat16):
    nheads = model_dim // headdim
    q = torch.randn(batch_size, seqlen, nheads, headdim, device="cuda", dtype=gen_dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    dtype = torch.float8_e4m3fn if "fp8" in provider else torch.bfloat16
    q, k, v = q.to(dtype).requires_grad_(), k.to(dtype).requires_grad_(), v.to(dtype).requires_grad_()
    
    # Quantization?
    q_descale = torch.randn((batch_size, nheads), device="cuda", dtype=torch.float32)
    k_descale = torch.randn((batch_size, nheads), device="cuda", dtype=torch.float32)
    v_descale = torch.randn((batch_size, nheads), device="cuda", dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if "flashattn" in provider:
        # Input is (B, N, H, D)
        if provider == "flashattn2-bf16":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash2(q, k, v, dropout_p, causal=False), quantiles=quantiles, rep=500)
        elif provider == "flashattn3-bf16":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash3(q, k, v, causal=False), quantiles=quantiles, rep=500)
        elif provider == "flashattn3-fp8":
            assert q.dtype in [torch.float8_e5m2, torch.float8_e4m3fn]
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash3(q, k, v, causal=False, q_descale=q_descale, k_descale=k_descale, v_descale=v_descale), quantiles=quantiles, rep=500)
        else:
            raise ValueError(f"FlashAttn Invalid provider: {provider}")
    elif "triton" in provider:
        # Input is (B, H, N, D)
        q, k, v = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
        if "fp8" in provider:
            v = v.transpose(-2, -1).contiguous().transpose(-2, -1)
            
        if provider == "triton-bf16":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: attention_forward(q, k, v, causal=False, sm_scale=1.0 / math.sqrt(headdim)), quantiles=quantiles, rep=500)
        elif provider == "triton-bf16-tma":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: tma_attention_forward(q, k, v, causal=False, sm_scale=1.0 / math.sqrt(headdim)), quantiles=quantiles, rep=500)
        elif provider == "triton-fp8":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: attention_forward(q, k, v, causal=False, sm_scale=1.0 / math.sqrt(headdim)), quantiles=quantiles, rep=500)
        elif provider == "triton-fp8-tma":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: tma_attention_forward(q, k, v, causal=False, sm_scale=1.0 / math.sqrt(headdim)), quantiles=quantiles, rep=500)
        else:
            raise ValueError(f"Triton Invalid provider: {provider}")
    elif "flashinfer" in provider:
        # Input is (N, H, D)
        def reshape_func(tensor):
            return tensor.permute(1, 0, 2, 3).reshape(seqlen, batch_size * nheads, headdim).contiguous()
        q, k, v = reshape_func(q), reshape_func(k), reshape_func(v)
        
        def reshape_scale_func(tensor):
            return tensor.flatten().contiguous()
        q_descale, k_descale, v_descale = reshape_scale_func(q_descale), reshape_scale_func(k_descale), reshape_scale_func(v_descale)
        
        if provider == "flashinfer2-bf16":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False, return_lse=False, backend="fa2"), quantiles=quantiles, rep=500)
        elif provider == "flashinfer3-bf16":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False, return_lse=False, backend="fa3"), quantiles=quantiles, rep=500)
        elif provider == "flashinfer2-fp8":
            raise NotImplementedError("FlashInfer2-FP8 is not supported in FlashInfer")
        elif provider == "flashinfer3-fp8":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False, return_lse=False, backend="fa3", scale_q=q_descale, scale_k=k_descale, scale_v=v_descale), quantiles=quantiles, rep=500)
        else:
            raise ValueError(f"FlashInfer Invalid provider: {provider}")
    else:
        raise ValueError(f"General Invalid provider: {provider}")

    flops = compute_flops(batch_size, nheads, seqlen, seqlen, headdim, headdim, causal=False)
    tflops = flops / (ms * 1e9)
    return tflops


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark different linear layer implementations')
    parser.add_argument(
        '--methods', 
        nargs='+', 
        default=['flashattn2-bf16', 'flashattn3-bf16'],
        help='Methods to benchmark.'
    )
    args = parser.parse_args()

    # Filter configs based on selected methods
    filtered_configs = []
    filtered_line_vals = []
    filtered_line_names = []
    filtered_styles = []
    
    if args.methods ==["all"]:
        args.methods = list(BENCH_CONFIGS.keys())
    for method in args.methods:
        filtered_line_vals.append(BENCH_CONFIGS[method]["line_vals"])
        filtered_line_names.append(BENCH_CONFIGS[method]["line_names"])
        filtered_styles.append(BENCH_CONFIGS[method]["color"])

    filtered_config = triton.testing.Benchmark(
        x_names=["seqlen"],  # X-axis: sequence length
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=filtered_line_vals,
        line_names=filtered_line_names,
        styles=filtered_styles,
        ylabel="TFLOPS",
        xlabel="Sequence Length",
        plot_name="flash_attn_perf",
        args={"batch_size": 2, "model_dim": 2048, "dropout_p": 0.0, "headdim": 128},
    )
    
    @triton.testing.perf_report(filtered_config)
    def benchmark(seqlen, batch_size, model_dim, dropout_p, headdim, provider):
        return benchmarker(seqlen, batch_size, model_dim, dropout_p, headdim, provider)
    
    benchmark.run(print_data=True)