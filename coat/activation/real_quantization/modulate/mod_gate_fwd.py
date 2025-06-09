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

from ..division._division_transpose import fp8_division_transpose
from ..common import FP8_MAX_VALUE, SCALE_MIN_THRES, get_configs_io_block

"""Element-wise Multiplication Forward"""
"""Input1 (Gate) uses 1 * 16 group quantization"""
"""Input2 (Up) uses 1 * 16 group quantization"""
"""Output uses per-tensor quantization"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""

fp8_max_value = {
    torch.float8_e4m3fn: 448,
    torch.float8_e5m2: 57344,
}


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
def fp8_mod_gate_pg2hp_forward_kernel(
    output_ptr,
    input1_ptr,
    input1_scale_ptr,  # input
    gate_ptr,
    M,
    N,
    SN,
    QB: tl.constexpr,
    fp8_max,  # shape
    input1_stride_0,
    input1_stride_1,  # input1 stride
    input1_stride_2,
    s_input1_stride_0,
    s_input1_stride_1,  # scale of input1 stride
    s_input1_stride_2,
    gate_stride_0,
    gate_stride_1,  # gate stride
    gate_stride_2,
    output_stride_0,
    output_stride_1,  # output stride
    output_stride_2,
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

    # --- The first input ---
    input1_block_ptr = tl.make_block_ptr(
        base=input1_ptr + pid_b * input1_stride_0,
        shape=(M, N),
        strides=(input1_stride_1, input1_stride_2),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # input ptr
    scale_input1_ptr = tl.make_block_ptr(
        base=input1_scale_ptr + pid_b * s_input1_stride_0,
        shape=(M, SN),
        strides=(s_input1_stride_1, s_input1_stride_2),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    input1 = tl.load(input1_block_ptr)
    scale_input1 = tl.load(scale_input1_ptr)
    input1 = input1.to(tl.float32)
    scale_input1 = scale_input1.to(tl.float32)

    # Dequantize and mul calculation
    scale_input1 = tl.reshape(scale_input1, (BLOCK_M, BLOCK_SN, 1))
    input1 = tl.reshape(input1, (BLOCK_M, BLOCK_SN, QB))
    input1 = input1 * scale_input1
    input1 = tl.reshape(input1, (BLOCK_M, BLOCK_N))

    # --- The second input ---
    gate_block_ptr = tl.make_block_ptr(
        base=gate_ptr + pid_b * gate_stride_0,
        shape=(1, N),
        strides=(gate_stride_1, gate_stride_2),
        offsets=(0, pid_dim1 * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )

    gate = tl.load(gate_block_ptr)
    gate = gate.to(tl.float32)
    gate = tl.reshape(gate, (1, BLOCK_N))

    # Actual Calculation of SiLU
    mod_output = input1 * gate
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


def fp8_mod_gate_pg2hp_forward(x1, s_x1, gate, QB, transpose_output_2d=False):
    """
    x1:        (..., M, N)
    s_x1:      (..., M, SN)
    gate:      (..., 1, N)
    y:         (..., M, N)
    """
    assert len(x1.shape) == 3

    # defining the input and output tensor
    B, M, N = x1.shape
    _, _, SN = s_x1.shape  # assume the shape of quantization block size is always 1 * G
    assert gate.shape == (B, 1, N)

    y = torch.empty_like(x1, dtype=torch.bfloat16)
    fp8MaxValue = fp8_max_value[x1.dtype]  # E4M3 and E5M2 have different max value

    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    fp8_mod_gate_pg2hp_forward_kernel[grid](
        y,
        x1,
        s_x1,
        gate,
        M,
        N,
        SN,
        QB,
        fp8MaxValue,
        x1.stride(0),
        x1.stride(1),
        x1.stride(2),
        s_x1.stride(0),
        s_x1.stride(1),
        s_x1.stride(2),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
    )

    return y
