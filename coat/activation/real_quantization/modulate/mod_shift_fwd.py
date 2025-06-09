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
# 4 block
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ..common import FP8_MAX_VALUE, SCALE_MIN_THRES, get_configs_io_block
from ..quantize._quantize_pertensor import fp8_quantize_pertensor

"""SiLU Activation Forward"""
"""Input uses 1 * 16 group quantization"""
"""Output uses 1 * 16 group quantization"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""


@triton.autotune(
    configs=[] + get_configs_io_block(),
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_SN": lambda args: args["BLOCK_N"] // args["QB"],
    }
)
@triton.jit
def _fp8_modulate_shift_pg2hp_forward_kernel(
    output_ptr,
    input_ptr,
    input_scale_ptr,  # input
    shift_ptr,
    scale_ptr,
    M,
    N,
    SN,
    QB: tl.constexpr,
    fp8_max,  # shape
    input_stride_0,
    input_stride_1,  # input stride
    input_stride_2,
    s_input_stride_0,
    s_input_stride_1,  # scale of input stride
    s_input_stride_2,
    output_stride_0,
    output_stride_1,  # output stride
    output_stride_2,
    shift_stride_0,
    shift_stride_1,
    shift_stride_2,
    scale_stride_0,
    scale_stride_1,
    scale_stride_2,
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):  # CUDA block size

    # Block PID
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid_t // NUM_BLOCK_N
    pid_dim1 = pid_t % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr + pid_b * input_stride_0,
        shape=(M, N),
        strides=(input_stride_1, input_stride_2),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # input ptr
    scale_input_ptr = tl.make_block_ptr(
        base=input_scale_ptr + pid_b * s_input_stride_0,
        shape=(M, SN),
        strides=(s_input_stride_1, s_input_stride_2),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )
    
    # shift ptr
    mod_shift_ptr = tl.make_block_ptr(
        base=shift_ptr + pid_b * shift_stride_0,
        shape=(1, N),
        strides=(shift_stride_1, shift_stride_2),
        offsets=(0, pid_dim1 * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    mod_scale_ptr = tl.make_block_ptr(
        base=scale_ptr + pid_b * scale_stride_0,
        shape=(1, N),
        strides=(scale_stride_1, scale_stride_2),
        offsets=(0, pid_dim1 * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr)
    scale_input = tl.load(scale_input_ptr)
    input = input.to(tl.float32)
    scale_input = scale_input.to(tl.float32)

    # Dequantize and silu calculation
    scale_input = tl.reshape(scale_input, (BLOCK_M, BLOCK_SN, 1))
    input = tl.reshape(input, (BLOCK_M, BLOCK_SN, QB))
    input = input * scale_input

    mod_shift = tl.load(mod_shift_ptr)
    mod_scale = tl.load(mod_scale_ptr)
    mod_shift = mod_shift.to(tl.float32)
    mod_scale = mod_scale.to(tl.float32)
    mod_scale = tl.reshape(mod_scale, (1, BLOCK_SN, QB))
    mod_shift = tl.reshape(mod_shift, (1, BLOCK_SN, QB))

    # Actual Calculation of Modulate
    mod_output = input * (1 + mod_scale) + mod_shift

    mod_output = mod_output.to(output_ptr.type.element_ty)
    mod_output = tl.reshape(mod_output, (BLOCK_M, BLOCK_N))

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + pid_b * output_stride_0,
        shape=(M, N),
        strides=(output_stride_1, output_stride_2),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    tl.store(output_block_ptr, mod_output)


def fp8_modulate_shift_pg2pt_forward(x, s_x, QB, shift, scale, return_transposed_2d=False):
    # The input should be 3D
    assert len(x.shape) == 3
    
    # defining the input and output tensor
    B, M, N = x.shape
    _, _, SN = s_x.shape  # assume the shape of quantization block size is always 1 * G
    assert shift.shape == (B, 1, N)
    assert scale.shape == (B, 1, N)

    y = torch.empty_like(x, dtype=x.dtype)
    s_y = torch.empty_like(s_x, dtype=s_x.dtype)
    fp8MaxValue = FP8_MAX_VALUE[x.dtype]  # E4M3 and E5M2 have different max value

    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    # We will do per-tensor quantization afterwards, so the output is high precision
    _fp8_modulate_shift_pg2hp_forward_kernel[grid](
        y,
        x,
        s_x,
        shift,
        scale,
        M,
        N,
        SN,
        QB,
        fp8MaxValue,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        s_x.stride(0),
        s_x.stride(1),
        s_x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        shift.stride(0),
        shift.stride(1),
        shift.stride(2),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
    )

    if return_transposed_2d:
        y, s_y_max, qy_t = fp8_quantize_pertensor(y, QB, x.dtype, return_transposed_2d=True)
        return y, s_y_max, qy_t
    else:
        y, s_y_max = fp8_quantize_pertensor(y, QB, x.dtype)
        return y, s_y_max
