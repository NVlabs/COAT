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

import unittest

import torch
import triton
from coat.activation.real_quantization.linear.linear import fp8matmul
from coat.activation.real_quantization.linear.fp8linear import FP8Linear
from coat.activation.deepseek.fp8linear import FP8DeepSeekLinear
from .test_utils import check_similarity, dequantize_tensor, quantize_tensor


def _test_linear_mse(input, weight, gradient, m, n, k, device):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Forward without Quantization
        bf16_input = input.clone().requires_grad_(True)
        bf16_linear = torch.nn.Linear(k, n, bias=False).to(torch.bfloat16)
        bf16_linear.weight.data = weight
        output_bf16_fwd = bf16_linear(bf16_input)

        # Forward + Quantization
        bf16_input_quant = input.clone().requires_grad_(True)
        bf16_linear_quant = FP8Linear(k, n, bias=False, device=device).to(torch.bfloat16)
        bf16_linear_quant.weight.data = weight
        output_bf16_quant_fwd = bf16_linear_quant(bf16_input_quant)

        # Forward + DeepSeek FP8
        fp8_input = input.clone().requires_grad_(True)
        fp8_linear = FP8DeepSeekLinear(k, n, bias=False, device=device).to(torch.bfloat16)
        fp8_linear.weight.data = weight
        output_fp8_fwd = fp8_linear(fp8_input)

        # Backward
        output_bf16_fwd.backward(gradient, retain_graph=True)
        output_bf16_quant_fwd.backward(gradient, retain_graph=True)
        output_fp8_fwd.backward(gradient, retain_graph=True)

        grad_input_bf16_bwd = bf16_input.grad.clone()
        grad_weight_bf16_bwd = bf16_linear.weight.grad.clone()
        grad_input_bf16_quant_bwd = bf16_input_quant.grad.clone()
        grad_weight_bf16_quant_bwd = bf16_linear_quant.weight.grad.clone()
        grad_input_fp8_bwd = fp8_input.grad.clone()
        grad_weight_fp8_bwd = fp8_linear.weight.grad.clone()

    return (
        output_bf16_fwd, output_bf16_quant_fwd, output_fp8_fwd, 
        grad_input_bf16_bwd, grad_input_bf16_quant_bwd, grad_input_fp8_bwd, 
        grad_weight_bf16_bwd, grad_weight_bf16_quant_bwd, grad_weight_fp8_bwd
    )
    
def _test_deepseek_linear(input, weight, gradient, m, n, k, device):
    bf16_input = input.clone().requires_grad_(True)
    fp8_input = input.clone().requires_grad_(True)
    
    # # Forward without Quantization
    # bf16_linear = torch.nn.Linear(k, n, bias=False).to(torch.bfloat16)
    # bf16_linear.weight.data = weight
    # output_bf16_fwd = bf16_linear(fp8_input)

    # Forward + Quantization
    bf16_linear = torch.nn.Linear(k, n, bias=False).to(torch.bfloat16)
    quant_input = quantize_tensor(input, 1, m, k, 128, torch.float8_e4m3fn, quant_type="per_group")
    quant_weight = quantize_tensor(weight, 1, n, k, 128, torch.float8_e4m3fn, quant_type="per_block")
    bf16_linear.weight.data = quant_weight[0]
    bf16_input = quant_input[0].squeeze(0).clone().detach().requires_grad_(True)
    output_bf16_fwd = bf16_linear(bf16_input)
    
    fp8_linear = FP8DeepSeekLinear(k, n, bias=False, device=device).to(torch.bfloat16)
    fp8_linear.weight.data = weight
    output_fp8_fwd = fp8_linear(fp8_input)

    # Backward
    output_bf16_fwd.backward(gradient)
    
    output_fp8_fwd.backward(gradient)

    grad_input_bf16_bwd = bf16_input.grad
    grad_weight_bf16_bwd = bf16_linear.weight.grad
    
    grad_input_fp8_bwd = fp8_input.grad
    grad_weight_fp8_bwd = fp8_linear.weight.grad
    
    return (
        output_bf16_fwd, output_fp8_fwd, 
        grad_input_bf16_bwd, grad_input_fp8_bwd, 
        grad_weight_bf16_bwd, grad_weight_fp8_bwd
    )


class TestLinear(unittest.TestCase):
    def test_linear(self):
        M, N, K = 4096, 2048, 8192
        QB = 16
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)

        scale_a, scale_b = torch.randn((1), device="cuda", dtype=torch.bfloat16), torch.randn(
            (1), device="cuda", dtype=torch.bfloat16
        )

        # Prepare row major and col-major data
        a = a.to(torch.float8_e4m3fn)
        b = b.T
        b = b.to(torch.float8_e4m3fn)

        a_32, b_32 = a.to(torch.float32), b.to(torch.float32)
        scale_ab = scale_a.to(torch.float32) * scale_b.to(torch.float32)
        output_torch = torch.matmul(a_32, b_32) * scale_ab
        output_torch_quantized, _, __ = quantize_tensor(output_torch.unsqueeze(0), 1, M, N, QB, torch.float8_e4m3fn)
        output_torch = output_torch.to(torch.bfloat16)

        # Output is not quantized
        output_fp8 = fp8matmul(a, b, False, scale_a, scale_b, QB)  # a should be row-major, b should be col-major

        # Output is quantized
        output_fp8_qx, output_fp8_sx = fp8matmul(
            a, b, True, scale_a, scale_b, QB
        )  # a should be row-major, b should be col-major
        output_fp8_rqx = dequantize_tensor(
            output_fp8_qx.unsqueeze(0), output_fp8_sx.unsqueeze(0), 1, M, N, QB
        )  # dequantize per-tensor quantization

        # TODO: This is not ideal
        # self.assertTrue(torch.allclose(output_torch, output_fp8, 1e-1, 0.2))
        # self.assertTrue(torch.allclose(output_torch_quantized, output_fp8_rqx, 1e-1, 0.5))

    def test_deepseek_linear(self):
        M, N, K = 4096, 2048, 8192
        device, QB = "cuda", 128
        
        a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
        b = torch.randn((N, K), device=device, dtype=torch.bfloat16)
        g = torch.randn((M, N), device=device, dtype=torch.bfloat16)
        
        output_bf16_fwd, output_fp8_deepseek_fwd, grad_input_bf16_bwd, grad_input_fp8_deepseek_bwd, grad_weight_bf16_bwd, grad_weight_fp8_deepseek_bwd = _test_deepseek_linear(a, b, g, M, N, K, device)
        
    def test_linear_mse(self):
        M, N, K = 16384, 16384, 16384
        device, QB = "cuda", 128
        
        a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
        b = torch.randn((N, K), device=device, dtype=torch.bfloat16)
        g = torch.randn((M, N), device=device, dtype=torch.bfloat16)
        
        (
            output_bf16_fwd, output_bf16_quant_fwd, output_fp8_fwd, \
            grad_input_bf16_bwd, grad_input_bf16_quant_bwd, grad_input_fp8_bwd, \
            grad_weight_bf16_bwd, grad_weight_bf16_quant_bwd, grad_weight_fp8_bwd
        ) = _test_linear_mse(a, b, g, M, N, K, device)

        # Calculate MSE
        mse_input_quant = torch.nn.MSELoss()(output_bf16_fwd, output_bf16_quant_fwd)
        mse_input_fp8 = torch.nn.MSELoss()(output_bf16_fwd, output_fp8_fwd)
        mse_grad_input_quant = torch.nn.MSELoss()(grad_input_bf16_bwd, grad_input_bf16_quant_bwd)
        mse_grad_input_fp8 = torch.nn.MSELoss()(grad_input_bf16_bwd, grad_input_fp8_bwd)
        mse_grad_weight_quant = torch.nn.MSELoss()(grad_weight_bf16_bwd, grad_weight_bf16_quant_bwd)
        mse_grad_weight_fp8 = torch.nn.MSELoss()(grad_weight_bf16_bwd, grad_weight_fp8_bwd)
        
        # Print MSEs
        print(f"MSE Input Quant: {mse_input_quant}")
        print(f"MSE Input FP8: {mse_input_fp8}")
        print(f"MSE Grad Input Quant: {mse_grad_input_quant}")
        print(f"MSE Grad Input FP8: {mse_grad_input_fp8}")
        print(f"MSE Grad Weight Quant: {mse_grad_weight_quant}")
        print(f"MSE Grad Weight FP8: {mse_grad_weight_fp8}")

        
if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200)
    unittest.main()
