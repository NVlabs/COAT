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

import torch
import triton

from coat.activation.real_quantization.act.gelu_fwd import fp8_gelu_forward
from coat.activation.real_quantization.quantize._quantize import fp8_quantize


def benchmarker(M, N, provider, groupsize: int = 16, dtype = torch.float8_e4m3fn):

    a = torch.randn((M, N), device="cuda", dtype=torch.bfloat16)
    a8, scale_a = fp8_quantize(a, groupsize, dtype)

    def torch_bf16():
        gelu_func = torch.nn.GELU()
        output_bf16 = gelu_func(a)

    def triton_fp8():
        _ = fp8_gelu_forward(a8, scale_a, groupsize, )

    def triton_fp8_output_transposed():
        _ = fp8_gelu_forward(a8, scale_a, groupsize, transpose_output_2d=True)
        
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-bf16":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_bf16(), quantiles=quantiles)
    elif provider == "triton-fp8":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fp8(), quantiles=quantiles)
    elif provider == "triton-fp8-output-transposed":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fp8_output_transposed(), quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N"],
            x_vals=[512 * i for i in range(2, 17)],
            line_arg="provider",
            line_vals=["torch-bf16", "triton-fp8", "triton-fp8-output-transposed"],
            line_names=["Torch-BF16", "Triton-fp8", "Triton-Fp8-Output-Transposed"],
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",
            plot_name="gelu-performance",
            args={"groupsize": 16},
        )
    )

    @triton.testing.perf_report(configs)
    def benchmark(M, N, groupsize, provider):
        return benchmarker(M, N, provider, groupsize)

    # def benchmark(M, provider):
    #     return benchmarker(M, 4096, 4096, 128, provider)

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200, precision=6)
    benchmark.run(print_data=True)
