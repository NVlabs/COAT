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

import os
import time
from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

from ..utils import quant_get_local_rank
from ..real_quantization.quantize._quantize import fp8_quantize
from ..real_quantization.quantize._quantize_transpose import fp8_quantize_transpose
from ..real_quantization.quantize._quantize_perblock import fp8_quantize_perblock
from ..real_quantization.quantize._quantize_perblock_transpose import fp8_quantize_perblock_transpose
from ..real_quantization.quantize._quantize_perblock_transpose_pergroup import fp8_quantize_perblock_transpose_pergroup
from .linear import fp8_deepseek_linear_backward, fp8_deepseek_linear_forward

import deep_gemm

@dataclass
class DefaultArgs:
    fabit: str = "E4M3"
    fwbit: str = "E4M3"
    bobit: str = "E4M3"


class FP8DeepSeekLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, args=None, layer_idx=0):
        super().__init__(in_features, out_features, bias, device)

        if args is None:  # I do not want to pass a new argument to OLMo so just use this method
            args = DefaultArgs()
        self.args = deepcopy(args)

        if quant_get_local_rank() == 0:
            print(f"[qlinear debug] Apply QLinear, {layer_idx}")

        self.layer_idx = layer_idx
        self.layer_name = None

    def forward(self, Input):
        if self.training:
            # if False:
            output = _FP8DeepSeekLinear.apply(Input, self.weight, self.bias, self.args, self.layer_name)
        else:
            output = F.linear(Input, self.weight, self.bias)

        return output

class _FP8DeepSeekLinear(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, bias, args, layer_name):
        (Qinput, Iscale), (Qinput_t, ITscale) = fp8_quantize_transpose(input, 128, args.fabit, scale_dtype=torch.float32)
        # To match deepseek's kernel requirement
        Iscale = Iscale.t().contiguous().t()
        ITscale = ITscale.t().contiguous().t()

        (Qweight, Wscale) = fp8_quantize_perblock(weight, 128, args.fwbit, scale_dtype=torch.float32)

        ctx.save_for_backward(Qinput_t, ITscale, weight, bias)
        ctx.utils = args, layer_name, input.shape, weight.shape
        
        fc_output = fp8_deepseek_linear_forward(Qinput, Iscale, Qweight, Wscale, 0, bias)

        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0]
        
        fc_output = fc_output.view(output_shape)

        return fc_output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        Qinput_t, ITscale, weight, bias = ctx.saved_tensors
        args, layer_name, input_shape, weight_shape = ctx.utils
        grad_output_shape = grad_output.shape
        
        # Flatten the gradient
        grad_output = grad_output.view(-1, grad_output.shape[-1])

        # For DGrad
        (Qgrad_output_pb, Gscale_pb), (Qgrad_output_t, Gscale_t), (Qgrad_output_pg, Gscale_pg) = \
            fp8_quantize_perblock_transpose_pergroup(grad_output, 128, args.bobit, scale_dtype=torch.float32, only_transposed=True)

        Gscale_pg = Gscale_pg.t().contiguous().t()

        Qweight_t, Wscale_t = fp8_quantize_perblock_transpose(weight, 128, args.fwbit, scale_dtype=torch.float32, only_transposed=True)

        grad_input, grad_weight = fp8_deepseek_linear_backward(
            Qinput_t,
            ITscale,
            Qgrad_output_pg,
            Gscale_pg,
            Qgrad_output_t,
            Gscale_t,
            Qweight_t,
            Wscale_t,
        )

        if bias is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        else:
            grad_bias = None

        grad_input = grad_input.view(input_shape)

        return grad_input, grad_weight, grad_bias, None, None