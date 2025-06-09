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

# PASS the unit test

import unittest

import torch
from test_utils import (check_similarity, dequantize_tensor, quantize_tensor,
                        random_tensor_generator)

from coat.activation.real_quantization.common import FP8_MAX_VALUE
from coat.activation.real_quantization.activation.gelu_bwd import fp8_gelu_backward
from coat.activation.real_quantization.activation.gelu_bwd_legacy import \
    fp8_gelu_backward_legacy
from coat.activation.real_quantization.act.gelu_fwd import fp8_gelu_forward
from coat.activation.real_quantization.act.mul_bwd import fp8_mul_backward
from coat.activation.real_quantization.act.mul_bwd_legacy import \
    fp8_mul_backward_legacy
from coat.activation.real_quantization.act.mul_pgpg2pt_fwd import fp8_mul_pgpg2pt_forward
from coat.activation.real_quantization.act.silu_bwd import fp8_silu_backward
from coat.activation.real_quantization.act.silu_bwd_legacy import \
    fp8_silu_backward_legacy
from coat.activation.real_quantization.act.silu_fwd import fp8_silu_forward


def _test_gelu_fwd(x, qx, sx, BS, SL, CDIM, QB):
    """
    Tests the forward pass of the GELU activation function,
    comparing the results between standard PyTorch and a custom FP8-based implementation.

    Returns:
    - output_torch (torch.Tensor): GELU output computed using PyTorch.
                                   We also quantize the output since we use fp8 precision flow.
    - output_triton (torch.Tensor): GELU output computed using FP8-based implementation.
    """
    torch_gelu = torch.nn.GELU()
    output_torch = torch_gelu(x)
    output_torch, _, _ = quantize_tensor(output_torch, BS, SL, CDIM, QB, qx.dtype, quant_type="per_block")

    x_triton, s_triton, x_t_triton = fp8_gelu_forward(qx, sx, QB)
    output_triton = dequantize_tensor(x_triton, s_triton, BS, SL, CDIM, QB)

    return output_torch, output_triton


def _test_gelu_bwd(x, qx, sx, g, BS, SL, CDIM, QB, fp8type):
    """
    Tests the backward pass of the GELU activation function,
    The output is per-tensor quantized, since it should follows a linear layer.
    """
    x = x.clone().requires_grad_(True)

    torch_gelu = torch.nn.GELU()
    output_torch = torch_gelu(x)
    output_torch.backward(g)
    grad_torch, _, __ = quantize_tensor(
        x.grad, BS, SL, CDIM, QB, fp8type, quant_type="per_block"
    )  # .to(bfloat16) since gradients are of bfloat16

    g_triton, s_triton = fp8_gelu_backward(qx, sx, g, QB, fp8type)
    grad_triton = dequantize_tensor(
        g_triton.unsqueeze(0), s_triton.unsqueeze(0), 1, SL, CDIM, QB
    )  # dequantize per-tensor quantization

    return grad_torch, grad_triton


def _test_gelu_bwd_legacy(x, qx, sx, g, qg, sg, BS, SL, CDIM, QB):
    """
    Tests the backward pass of the GELU activation function,
    The output is per-tensor quantized, since it should follows a linear layer.
    """
    x = x.clone().requires_grad_(True)

    torch_gelu = torch.nn.GELU()
    output_torch = torch_gelu(x)
    output_torch.backward(g)
    grad_torch, _, __ = quantize_tensor(
        x.grad, BS, SL, CDIM, QB, qg.dtype, quant_type="per_block"
    )  # .to(bfloat16) since gradients are of bfloat16

    g_triton, s_triton = fp8_gelu_backward_legacy(qx, sx, qg, sg, QB)
    grad_triton = dequantize_tensor(g_triton, s_triton, BS, SL, CDIM, QB)  # dequantize per-tensor quantization

    return grad_torch, grad_triton


def _test_silu_fwd(x, qx, sx, BS, SL, CDIM, QB):
    """
    Tests the forward pass of the SiLU activation function,
    """
    torch_silu = torch.nn.SiLU()
    output_torch = torch_silu(x)
    output_torch, _, _ = quantize_tensor(output_torch, BS, SL, CDIM, QB, qx.dtype)

    x_triton, s_triton = fp8_silu_forward(qx, sx, QB)
    output_triton = dequantize_tensor(x_triton, s_triton, BS, SL, CDIM, QB)
    return output_torch, output_triton


def _test_silu_bwd(x, qx, sx, g, BS, SL, CDIM, QB, fp8type):
    """
    Tests the backward pass of the SiLU activation function,
    The output is per-tensor full precision, since it should follows a linear layer. We do not quantize it right not, and require to quantize it afterwards.
    """
    x = x.clone().requires_grad_(True)

    torch_silu = torch.nn.SiLU()
    output_torch = torch_silu(x)
    output_torch.backward(g)
    grad_torch = x.grad.to(torch.bfloat16)
    grad_torch, _, __ = quantize_tensor(
        x.grad, BS, SL, CDIM, QB, fp8type, quant_type="per_block"
    )  # .to(bfloat16) since gradients are of bfloat16

    grad_triton, _ = fp8_silu_backward(qx, sx, g, QB, fp8type)

    return grad_torch, grad_triton


def _test_silu_bwd_legacy(x, qx, sx, g, qg, sg, BS, SL, CDIM, QB):
    """
    Tests the backward pass of the SiLU activation function,
    The output is per-tensor full precision, since it should follows a linear layer. We do not quantize it right not, and require to quantize it afterwards.
    """
    x = x.clone().requires_grad_(True)

    torch_silu = torch.nn.SiLU()
    output_torch = torch_silu(x)
    output_torch.backward(g)
    grad_torch = x.grad.to(torch.bfloat16)

    grad_triton, _ = fp8_silu_backward_legacy(qx, sx, qg, sg, QB)

    return grad_torch, grad_triton


def _test_mul_fwd(x1, qx1, sx1, x2, qx2, sx2, BS, SL, CDIM, QB):
    """
    Tests the forward pass of the Element-wise Multiplication function,
    X1 represents Gate Proj, and X2 represents Up Proj.
    The output is per-tensor quantized, since it should follows a linear layer.
    """
    output_torch = x1 * x2
    output_torch, _, _ = quantize_tensor(output_torch, BS, SL, CDIM, QB, qx1.dtype, quant_type="per_block")

    x_triton, s_triton, x_t_triton = fp8_mul_pgpg2pt_forward(qx1, sx1, qx2, sx2, QB)
    output_triton = dequantize_tensor(x_triton, s_triton, BS, SL, CDIM, QB)

    return output_torch, output_triton


def _test_mul_bwd(x1, qx1, sx1, x2, qx2, sx2, g, BS, SL, CDIM, QB, fp8type):
    """
    Tests the backward pass of the Element-wise Multiplication function,
    We return the gradient of Gate Proj and Up Proj. G1 is per-group quantized, and G2 is per-tensor quantized.
    """
    x1 = x1.clone().requires_grad_(True)
    x2 = x2.clone().requires_grad_(True)
    output_torch = x1 * x2
    output_torch.backward(g)
    output_torch1 = x1.grad.to(torch.bfloat16)
    output_torch2, _, __ = quantize_tensor(x2.grad, BS, SL, CDIM, QB, fp8type, quant_type="per_block")

    g1_triton, (output_triton2, _) = fp8_mul_backward(qx1, sx1, qx2, sx2, g, QB, fp8type)
    output_triton1 = g1_triton

    return output_torch1, output_triton1, output_torch2, output_triton2


def _test_mul_bwd_legacy(x1, qx1, sx1, x2, qx2, sx2, g, qg, sg, BS, SL, CDIM, QB):
    """
    Tests the backward pass of the Element-wise Multiplication function,
    We return the gradient of Gate Proj and Up Proj. G1 is per-group quantized, but G2 is per-tensor full precision.
    """
    x1 = x1.clone().requires_grad_(True)
    x2 = x2.clone().requires_grad_(True)
    output_torch = x1 * x2
    output_torch.backward(g)
    output_torch1, _, __ = quantize_tensor(
        x1.grad, BS, SL, CDIM, QB, qg.dtype
    )  # NOTE: qg.dtype is e5m2, while qx uses e4m3
    output_torch2 = x2.grad.to(torch.bfloat16)

    g1_triton, s1_triton, g2_triton, s2_triton = fp8_mul_backward_legacy(qx1, sx1, qx2, sx2, qg, sg, QB)
    output_triton1 = dequantize_tensor(g1_triton, s1_triton, BS, SL, CDIM, QB)
    output_triton2 = g2_triton

    return output_torch1, output_triton1, output_torch2, output_triton2


class TestActivation(unittest.TestCase):
    """
    x: reconstructed activation (quantize it then dequantize)
    qx: quantized activation
    sx: scale factor of activation

    g: reconstrcted gradient (quantize it then dequantize)
    qg: quantized gradient
    sg: scale factor of gradient
    """

    # Test the forward of GELU
    def test_gelu_fwd(self):
        BS, SL, CDIM, QB, fp8type = [4, 256, 2048, 16, torch.float8_e4m3fn]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        output_torch, output_triton = _test_gelu_fwd(x, qx, sx, BS, SL, CDIM, QB)

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch, output_triton))

    # Test the backward of GELU
    def test_gelu_bwd(self):
        BS, SL, CDIM, QB, fp8type_act, fp8type_grad = [4, 256, 2048, 16, torch.float8_e4m3fn, torch.float8_e5m2]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type_act)
        g, _, _ = random_tensor_generator(BS, SL, CDIM, QB, fp8type_grad)

        output_torch, output_triton = _test_gelu_bwd(x, qx, sx, g, BS, SL, CDIM, QB, fp8type_grad)

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        import IPython; IPython.embed()
        self.assertTrue(check_similarity(output_torch, output_triton))

    # Test the backward of GELU
    def test_gelu_bwd_legacy(self):
        BS, SL, CDIM, QB, fp8type_act, fp8type_grad = [4, 256, 2048, 16, torch.float8_e4m3fn, torch.float8_e5m2]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type_act)
        g, qg, sg = random_tensor_generator(BS, SL, CDIM, QB, fp8type_grad)
        output_torch, output_triton = _test_gelu_bwd_legacy(x, qx, sx, g, qg, sg, BS, SL, CDIM, QB)

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch, output_triton))

    # Test the forward of SiLU
    def test_silu_fwd(self):
        BS, SL, CDIM, QB, fp8type = [4, 256, 2048, 16, torch.float8_e4m3fn]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        output_torch, output_triton = _test_silu_fwd(x, qx, sx, BS, SL, CDIM, QB)

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch, output_triton))

    # Test the backward of SiLU
    def test_silu_bwd(self):
        BS, SL, CDIM, QB, fp8type_act, fp8type_grad = [4, 256, 2048, 16, torch.float8_e4m3fn, torch.float8_e5m2]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type_act)
        g, _, _ = random_tensor_generator(BS, SL, CDIM, QB, fp8type_grad)
        output_torch, output_triton = _test_silu_bwd(x, qx, sx, g, BS, SL, CDIM, QB, fp8type_grad)

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch, output_triton))

    # Test the backward of SiLU
    def test_silu_bwd_legacy(self):
        BS, SL, CDIM, QB, fp8type_act, fp8type_grad = [4, 256, 2048, 16, torch.float8_e4m3fn, torch.float8_e5m2]
        x, qx, sx = random_tensor_generator(BS, SL, CDIM, QB, fp8type_act)
        g, qg, sg = random_tensor_generator(BS, SL, CDIM, QB, fp8type_grad)
        output_torch, output_triton = _test_silu_bwd_legacy(x, qx, sx, g, qg, sg, BS, SL, CDIM, QB)

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch, output_triton))

    # Test the forward of element-wise multiplication
    def test_mul_fwd(self):
        BS, SL, CDIM, QB, fp8type = [4, 256, 2048, 16, torch.float8_e4m3fn]
        x1, qx1, sx1 = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        x2, qx2, sx2 = random_tensor_generator(BS, SL, CDIM, QB, fp8type)
        output_torch, output_triton = _test_mul_fwd(x1, qx1, sx1, x2, qx2, sx2, BS, SL, CDIM, QB)

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch, output_triton))

    # Test the backward of element-wise multiplication
    def test_mul_bwd(self):
        BS, SL, CDIM, QB, fp8type_act, fp8type_grad = [4, 256, 2048, 16, torch.float8_e4m3fn, torch.float8_e5m2]
        x1, qx1, sx1 = random_tensor_generator(BS, SL, CDIM, QB, fp8type_act)
        x2, qx2, sx2 = random_tensor_generator(BS, SL, CDIM, QB, fp8type_act)
        g, _, _ = random_tensor_generator(BS, SL, CDIM, QB, fp8type_grad)
        output_torch1, output_triton1, output_torch2, output_triton2 = _test_mul_bwd(
            x1, qx1, sx1, x2, qx2, sx2, g, BS, SL, CDIM, QB, fp8type_grad
        )

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch1, output_triton1))
        self.assertTrue(check_similarity(output_torch2, output_triton2))

    # Test the backward of element-wise multiplication
    def test_mul_bwd_legacy(self):
        BS, SL, CDIM, QB, fp8type_act, fp8type_grad = [4, 256, 2048, 16, torch.float8_e4m3fn, torch.float8_e5m2]
        x1, qx1, sx1 = random_tensor_generator(BS, SL, CDIM, QB, fp8type_act)
        x2, qx2, sx2 = random_tensor_generator(BS, SL, CDIM, QB, fp8type_act)
        g, qg, sg = random_tensor_generator(BS, SL, CDIM, QB, fp8type_grad)
        output_torch1, output_triton1, output_torch2, output_triton2 = _test_mul_bwd_legacy(
            x1, qx1, sx1, x2, qx2, sx2, g, qg, sg, BS, SL, CDIM, QB
        )

        # This is a very loose assert. Most values should be the same.
        # self.assertTrue(torch.allclose(output_torch + 1e-8, output_triton + 1e-8, 1e-2, 0.2))
        self.assertTrue(check_similarity(output_torch1, output_triton1))
        self.assertTrue(check_similarity(output_torch2, output_triton2))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200)
    unittest.main()
