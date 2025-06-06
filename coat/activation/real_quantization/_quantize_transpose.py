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

from .common import (FP8_MAX_VALUE, SCALE_MIN_THRES, convert_fp8_to_embit,
                     convert_str_to_fp8, get_tuned_kernel)

"""Quantize Operator"""
"""Input uses 1 * 16 group quantization"""
"""Output uses 1 * 16 group quantization"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""


@triton.heuristics(
    {
        "BLOCK_SM": lambda args: args["BLOCK_M"] // args["QB"],
        "BLOCK_SN": lambda args: args["BLOCK_N"] // args["QB"],
    }
)
@triton.jit
def _fp8_quantize_transpose_kernel(
    output_ptr,
    output_t_ptr,
    output_scale_ptr,  # output
    output_scale_t_ptr,  # output
    input_ptr,  # input
    M,
    N,
    SM,
    SN,
    QB: tl.constexpr,
    fp8_max,  # shape
    input_stride_0,
    input_stride_1,  # input stride
    output_stride_0,
    output_stride_1,  # output stride
    output_t_stride_0,
    output_t_stride_1,  # output stride
    s_output_stride_0,
    s_output_stride_1,  # scale of output stride
    s_output_t_stride_0,
    s_output_t_stride_1,  # scale of output stride
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SM: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):  # CUDA block size

    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr, boundary_check=(0, 1))
    input = input.to(tl.float32)

    # ========== Start Per-Group Quant ==========
    output = tl.reshape(input, (BLOCK_M, BLOCK_SN, QB))

    # Quantize Scale calculation
    abs_output = tl.abs(output)
    max_val = tl.max(abs_output, axis=2) + SCALE_MIN_THRES
    scale_output = max_val / fp8_max
    scale_output = tl.reshape(scale_output, (BLOCK_M, BLOCK_SN, 1))

    # Quantize
    output = tl.fdiv(output, scale_output)

    output = output.to(output_ptr.type.element_ty)

    scale_output = scale_output.to(output_scale_ptr.type.element_ty)
    scale_output = tl.reshape(scale_output, (BLOCK_M, BLOCK_SN))
    output = tl.reshape(output, (BLOCK_M, BLOCK_N))

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    scale_output_ptr = tl.make_block_ptr(
        base=output_scale_ptr,
        shape=(M, SN),
        strides=(s_output_stride_0, s_output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    tl.store(output_block_ptr, output, boundary_check=(0, 1))
    tl.store(scale_output_ptr, scale_output, boundary_check=(0, 1))

    # ========== End Per-Group Quant ==========

    # ========== Start Transposed Per-Group Quant ==========
    input = tl.trans(input)
    output = tl.reshape(input, (BLOCK_N, BLOCK_SM, QB))

    # Quantize Scale calculation
    abs_output = tl.abs(output)
    max_val = tl.max(abs_output, axis=2) + SCALE_MIN_THRES
    scale_output = max_val / fp8_max
    scale_output = tl.reshape(scale_output, (BLOCK_N, BLOCK_SM, 1))

    # Quantize
    output = tl.fdiv(output, scale_output)

    output = output.to(output_t_ptr.type.element_ty)

    scale_output = scale_output.to(output_scale_t_ptr.type.element_ty)
    scale_output = tl.reshape(scale_output, (BLOCK_N, BLOCK_SM))
    output = tl.reshape(output, (BLOCK_N, BLOCK_M))

    # pointers
    output_t_block_ptr = tl.make_block_ptr(
        base=output_t_ptr,
        shape=(N, M),
        strides=(output_t_stride_0, output_t_stride_1),
        offsets=(pid_dim1 * BLOCK_N, pid_dim0 * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0),
    )
    scale_output_t_ptr = tl.make_block_ptr(
        base=output_scale_t_ptr,
        shape=(N, SM),
        strides=(s_output_t_stride_0, s_output_t_stride_1),
        offsets=(pid_dim1 * BLOCK_N, pid_dim0 * BLOCK_SM),
        block_shape=(BLOCK_N, BLOCK_SM),
        order=(1, 0),
    )

    tl.store(output_t_block_ptr, output, boundary_check=(0, 1))
    tl.store(scale_output_t_ptr, scale_output, boundary_check=(0, 1))

    # ========== End Transposed Per-Group Quant ==========
    
_fp8_quantize_transpose_kernel_tuned = {
    QB: get_tuned_kernel(_fp8_quantize_transpose_kernel, QB=QB)
    for QB in [16, 32, 64, 128]
}

def fp8_quantize_transpose(x, QB, fp8type, scale_dtype=torch.bfloat16):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    SM, SN = M // QB, N // QB

    if isinstance(fp8type, str):
        fp8type = convert_str_to_fp8[fp8type]
    y = torch.empty_like(x, dtype=fp8type)
    y_t = torch.empty((N, M), dtype=fp8type, device=x.device)
    s_y = torch.empty((M, SN), dtype=scale_dtype, device=x.device)
    s_y_t = torch.empty((N, SM), dtype=scale_dtype, device=x.device)
    fp8MaxValue = FP8_MAX_VALUE[fp8type]  # E4M3 and E5M2 have different max value

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    # if x.shape[1] == 11008:
    #     import IPython
    #     IPython.embed()

    _fp8_quantize_transpose_kernel_tuned[QB][grid](
        y,
        y_t,
        s_y,
        s_y_t,
        x,
        M,
        N,
        SM,
        SN,
        QB,
        fp8MaxValue,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        y_t.stride(0),
        y_t.stride(1),
        s_y.stride(0),
        s_y.stride(1),
        s_y_t.stride(0),
        s_y_t.stride(1),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
    )

    return (y, s_y), (y_t, s_y_t)
