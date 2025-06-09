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

# Pass unit test

import unittest

import torch
from .test_utils import (check_similarity, dequantize_tensor, quantize_tensor,
                        random_tensor_generator)

from coat.activation.real_quantization.norm.func_layernorm_noparam import (
    fp8_layernorm_noparam_backward, fp8_layernorm_noparam_forward)
from coat.activation.real_quantization.norm.func_layernorm_param import (
    fp8_layernorm_param_backward, fp8_layernorm_param_forward)
from coat.activation.real_quantization.norm.func_rmsnorm import (
    fp8_rmsnorm_backward, fp8_rmsnorm_forward)


def _test_layer_norm_noparam(x, qx, sx, g, BS, SL, CDIM, QB, eps=1e-5):
    """
    Test the forward and backward of LayerNorm (no param)
    Forward: input is per-group quant, output is per-tensor quanti
    Backward: input and output are full precision
    """
    x = x.clone().requires_grad_(True)

    # Forward
    w_shape = (CDIM,)
    torch_layer_norm = torch.nn.functional.layer_norm
    _output_torch = torch_layer_norm(x, w_shape, eps=eps)
    output_torch_fwd, _, _ = quantize_tensor(_output_torch, BS, SL, CDIM, QB, qx.dtype, quant_type="per_block")

    x_triton, s_triton, x_t_triton, (mean, rstd, num_warps) = fp8_layernorm_noparam_forward(qx, sx, QB, eps)
    output_triton_fwd = dequantize_tensor(x_triton, s_triton, BS, SL, CDIM, QB)

    # Backward
    _output_torch.backward(g)
    output_torch_bwd = x.grad.to(torch.bfloat16)

    output_triton_bwd = fp8_layernorm_noparam_backward(qx, sx, g, QB, mean, rstd, num_warps)

    return output_torch_fwd, output_triton_fwd, output_torch_bwd, output_triton_bwd


def _test_layer_norm_param(x, qx, sx, g, w, b, BS, SL, CDIM, QB, eps=1e-5):
    """
    Test the forward and backward of LayerNorm (no param)
    Forward: input is per-group quant, output is per-tensor quanti
    Backward: input and output are full precision
    """
    x = x.clone().requires_grad_(True)

    # Forward
    w_shape = (CDIM,)
    torch_layer_norm = torch.nn.functional.layer_norm
    _output_torch = torch_layer_norm(x, w_shape, weight=w, bias=b, eps=eps)
    output_torch_fwd, _, _ = quantize_tensor(_output_torch, BS, SL, CDIM, QB, qx.dtype, quant_type="per_block")

    x_triton, s_triton, x_t_triton, (_, _, mean, rstd, num_warps) = fp8_layernorm_param_forward(qx, sx, w, b, QB, eps)
    output_triton_fwd = dequantize_tensor(x_triton, s_triton, BS, SL, CDIM, QB)

    # Backward
    _output_torch.backward(g)
    output_torch_bwd = x.grad.to(torch.bfloat16)
    output_w_torch_bwd = w.grad
    output_b_torch_bwd = b.grad

    output_triton_bwd, output_w_triton_bwd, output_b_triton_bwd = fp8_layernorm_param_backward(qx, sx, g, QB, w, b, mean, rstd, num_warps)

    return (
        output_torch_fwd, output_triton_fwd, 
        output_torch_bwd, output_triton_bwd,
        output_w_torch_bwd, output_w_triton_bwd,
        output_b_torch_bwd, output_b_triton_bwd,
    )


def _test_rms_norm(x, qx, sx, g, w, BS, SL, CDIM, QB, eps=1e-5):
    """
    Test the forward and backward of RMSNorm (no param)
    Forward: input is per-group quant, output is per-tensor quanti
    Backward: input and output are full precision
    """
    x = x.clone().requires_grad_(True)

    # Forward
    w_shape = (CDIM,)
    torch_rms_norm = torch.nn.functional.rms_norm
    _output_torch = torch_rms_norm(x, w_shape, w, eps=eps)
    output_torch_fwd, _, _ = quantize_tensor(_output_torch, BS, SL, CDIM, QB, qx.dtype, quant_type="per_block")

    x_triton, s_triton, x_t_triton, (_, rstd, num_warps) = fp8_rmsnorm_forward(qx, sx, w, QB, eps)
    output_triton_fwd = dequantize_tensor(x_triton, s_triton, BS, SL, CDIM, QB)

    # Backward
    _output_torch.backward(g)
    output_torch_bwd = x.grad.to(torch.bfloat16)
    output_w_torch_bwd = w.grad

    output_triton_bwd, output_w_triton_bwd = fp8_rmsnorm_backward(qx, sx, g, w, rstd, QB, num_warps)

    return (
        output_torch_fwd,
        output_triton_fwd,
        output_torch_bwd,
        output_triton_bwd,
        output_w_torch_bwd,
        output_w_triton_bwd,
    )


class TestNorm(unittest.TestCase):
    """
    x: reconstructed activation (quantize it then dequantize)
    qx: quantized activation
    sx: scale factor of activation

    g: reconstrcted gradient (quantize it then dequantize)
    qg: quantized gradient
    sg: scale factor of gradient
    """

    # Test the forward of LayerNorm
    def test_layernorm_noparam(self):
        BS, SL, CDIM, QB, fp8type = [4, 256, 2048, 16, torch.float8_e4m3fn]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        g, _, _ = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        output_torch_fwd, output_triton_fwd, output_torch_bwd, output_triton_bwd = _test_layer_norm_noparam(
            x, qx, sx, g, BS, SL, CDIM, QB
        )

    # Test the forward of LayerNorm
    def test_layernorm_param(self):
        BS, SL, CDIM, QB, fp8type = [4, 256, 2048, 16, torch.float8_e4m3fn]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        g, _, _ = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        w, b = [torch.rand((CDIM,), device="cuda", requires_grad=True) for _ in range(2)]
        (
            output_torch_fwd, output_triton_fwd, 
            output_torch_bwd, output_triton_bwd,
            output_w_torch_bwd, output_w_triton_bwd,
            output_b_torch_bwd, output_b_triton_bwd,
        ) = _test_layer_norm_param(x, qx, sx, g, w, b, BS, SL, CDIM, QB)

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch_fwd, output_triton_fwd))
        self.assertTrue(check_similarity(output_torch_bwd, output_triton_bwd))
        self.assertTrue(torch.allclose(output_w_torch_bwd + 1e-8, output_w_triton_bwd + 1e-8, 1e-3))
        self.assertTrue(torch.allclose(output_b_torch_bwd + 1e-8, output_b_triton_bwd + 1e-8, 1e-3))

    # Test the forward of RMSNorm
    def test_rmsnorm(self):
        BS, SL, CDIM, QB, fp8type = [4, 256, 2048, 16, torch.float8_e4m3fn]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        g, _, _ = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        w = torch.rand((CDIM,), device="cuda", requires_grad=True)
        (
            output_torch_fwd,
            output_triton_fwd,
            output_torch_bwd,
            output_triton_bwd,
            output_w_torch_bwd,
            output_w_triton_bwd,
        ) = _test_rms_norm(x, qx, sx, g, w, BS, SL, CDIM, QB)

        # This is a very loose assert. Most values should be the same.
        self.assertTrue(check_similarity(output_torch_fwd, output_triton_fwd))
        self.assertTrue(check_similarity(output_torch_bwd, output_triton_bwd))
        self.assertTrue(torch.allclose(output_w_torch_bwd + 1e-8, output_w_triton_bwd + 1e-8, 1e-3))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200)
    unittest.main()
