# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
import argparse

import torch
import triton

from coat.activation.real_quantization.linear.linear import fp8matmul

########################################################
#                  Linear Layer Functions              #
########################################################

def torch_bf16(a16, b16):
    output_bf16 = torch.matmul(a16, b16)
    
def torch_fp8_slow_acc(a, b, scale_a_32, scale_b_32, bias):
    """ A: (M, K), row-major | B: (K, N), col-major | scale_a: (1), float32 | scale_b: (1), float32 | bias: (N), float16 | output: (M, N), row-major """
    output_fp8 = torch._scaled_mm(a, b, scale_a_32, scale_b_32, bias=bias, out_dtype=torch.bfloat16, use_fast_accum=False)
    
def torch_fp8_fast_acc(a, b, scale_a_32, scale_b_32, bias):
    """ A: (M, K), row-major | B: (K, N), col-major | scale_a: (1), float32 | scale_b: (1), float32 | bias: (N), float16 | output: (M, N), row-major """
    output_fp8 = torch._scaled_mm(a, b, scale_a_32, scale_b_32, bias=bias, out_dtype=torch.bfloat16, use_fast_accum=True)
    
def torch_fp8_fast_acc_output_quantized(a, b, scale_a_32, scale_b_32, bias): # Not useful, because the output is quantized per-tensor
    """ A: (M, K), row-major | B: (K, N), col-major | scale_a: (1), float32 | scale_b: (1), float32 | bias: (N), float16 | output: (M, N), row-major """
    output_fp8 = torch._scaled_mm(a, b, scale_a_32, scale_b_32, bias=bias, out_dtype=torch.float8_e4m3fn, use_fast_accum=True)

def triton_fp8_output_fp(a, b, scale_a, scale_b, groupsize, bias):
    """ A: (M, K), row-major | B: (K, N), col-major | scale_a: (1) | scale_b: (1) | bias: (N) | output: (M, N), row-major """
    output_fp8_y = fp8matmul(a, b, False, scale_a, scale_b, groupsize, bias=bias)

def triton_fp8_output_quantized(a, b, scale_a, scale_b, groupsize, bias):
    """ A: (M, K), row-major | B: (K, N), col-major | scale_a: (1) | scale_b: (1) | bias: (N) | output: (M, N), row-major """
    output_fp8_y, output_fp8_s = fp8matmul(a, b, True, scale_a, scale_b, groupsize, bias=bias)
    
def deepseek_fp8_define(ds_x_fp8, ds_y_fp8, ds_out):
    """ A: (M, K) | B: (N, K) | scale_a: (M, K // QB) | scale_b: (N // QB, K // QB) | bias: (N) | output: (M, N) """
    import deep_gemm
    return deep_gemm.gemm_fp8_fp8_bf16_nt(ds_x_fp8, ds_y_fp8, ds_out)
    
########################################################
#                  Benchmark Functions                 #
########################################################

def benchmarker(M, N, K, provider, scalar, groupsize: int = 16):
    M, N, K = M * scalar[0], N * scalar[1], K * scalar[2]
    
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((N), device="cuda", dtype=torch.bfloat16)

    scale_a, scale_b = torch.randn((1), device="cuda", dtype=torch.float32), torch.randn((1), device="cuda", dtype=torch.float32)
    
    quantiles = [0.5, 0.2, 0.8]
    if "torch" in provider:
        a16, b16 = a.to(torch.bfloat16), b.T.to(torch.bfloat16)
        scale_a_32, scale_b_32 = scale_a.to(torch.float32), scale_b.to(torch.float32)
        
        if provider == "torch-bf16":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_bf16(a16, b16), quantiles=quantiles, rep=500)
        elif provider == "torch-fp8-slow-acc":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_fp8_slow_acc(a, b, scale_a_32, scale_b_32, bias), quantiles=quantiles, rep=500)
        elif provider == "torch-fp8-fast-acc":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_fp8_fast_acc(a, b, scale_a_32, scale_b_32, bias), quantiles=quantiles, rep=500)
        elif provider == "torch-fp8-fast-acc-output-quantized":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_fp8_fast_acc_output_quantized(a, b, scale_a_32, scale_b_32, bias), quantiles=quantiles, rep=500)
    elif "triton" in provider:
        a = a.to(torch.float8_e4m3fn)
        b = b.T
        b = b.to(torch.float8_e4m3fn)

        if provider == "triton-fp8-output-fp":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fp8_output_fp(a, b, scale_a, scale_b, groupsize, bias), quantiles=quantiles, rep=500)
        elif provider == "triton-fp8-output-quantized":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fp8_output_quantized(a, b, scale_a, scale_b, groupsize, bias), quantiles=quantiles, rep=500)
    elif "deepseek" in provider:
        if provider == "deepseek-fp8":    # DeepSeek's input
            QB = 128
            ds_scale_a = torch.rand((M, K // QB), dtype=torch.float32, device="cuda")
            ds_scale_b = torch.rand((N // QB, K // QB), dtype=torch.float32, device="cuda")
            ds_scale_a = ds_scale_a.t().contiguous().t()
            
            ds_x_fp8, ds_y_fp8 = (a, ds_scale_a), (b.t(), ds_scale_b)
            ds_out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

            ms, min_ms, max_ms = triton.testing.do_bench(lambda: deepseek_fp8_define(ds_x_fp8, ds_y_fp8, ds_out), quantiles=quantiles, rep=500)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


BENCH_CONFIGS = {
    "torch-bf16": {
        "line_vals": "torch-bf16",
        "line_names": "Torch-BF16",
        "color": ("green", "-"),
    },
    "torch-fp8-slow-acc": {
        "line_vals": "torch-fp8-slow-acc",
        "line_names": "Torch-FP8-Slow-Acc",
        "color": ("purple", "-"),
    },
    "torch-fp8-fast-acc": {
        "line_vals": "torch-fp8-fast-acc",
        "line_names": "Torch-FP8-Fast-Acc",
        "color": ("purple", "-"),
    },
    "torch-fp8-fast-acc-output-quantized": {
        "line_vals": "torch-fp8-fast-acc-output-quantized",
        "line_names": "Torch-FP8-Fast-Acc-Output-Quantized",
        "color": ("purple", "-"),
    },
    "triton-fp8-output-fp": {
        "line_vals": "triton-fp8-output-fp",
        "line_names": "Triton-FP8-Output-FP",
        "color": ("blue", "-"),
    },
    "triton-fp8-output-quantized": {
        "line_vals": "triton-fp8-output-quantized",
        "line_names": "Triton-FP8-Output-Quantized",
        "color": ("red", "-"),
    },
    "deepseek-fp8": {
        "line_vals": "deepseek-fp8",
        "line_names": "Deepseek-fp8",
        "color": ("orange", "-"),
    },
}

def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark different linear layer implementations')
    parser.add_argument(
        '--methods', 
        nargs='+', 
        default=['torch-bf16', 'triton-fp8-output-fp'],
        help='Methods to benchmark.'
    )
    parser.add_argument(
        '--scalar',
        type=parse_comma_separated_ints,
        default=(1, 2, 4),
        help='Multiply the size of the input by this factor. Order is M, N, K.'
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
        x_names=["M", "N", "K"],
        x_vals=[1024 * i for i in range(2, 17)],
        line_arg="provider",
        line_vals=filtered_line_vals,
        line_names=filtered_line_names,
        styles=filtered_styles,
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
    filtered_configs.append(filtered_config)

    @triton.testing.perf_report(filtered_configs)
    def benchmark(M, N, K, provider):
        return benchmarker(M, N, K, provider, args.scalar)

    # def benchmark(M, provider):
    #     return benchmarker(M, 4096, 4096, 128, provider)

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200, precision=6)
    print("Matrix Scalar: ", args.scalar)
    benchmark.run(print_data=True)
