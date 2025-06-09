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

from ...utils import quant_get_local_rank
from ..division._division_transpose import fp8_division_transpose
from ..quantize._quantize_pertensor_transpose import fp8_quantize_pertensor_transpose
from .linear import fp8_linear_backward, fp8_linear_forward


@dataclass
class DefaultArgs:
    fabit: str = "E4M3"
    fwbit: str = "E4M3"
    bobit: str = "E5M2"


class FP8Linear(nn.Linear):
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
            output = QuantLinearTE.apply(Input, self.weight, self.bias, self.args, self.layer_name)
        else:
            output = F.linear(Input, self.weight, self.bias)

        return output

class QuantLinearTE(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)
    def forward(ctx, input, weight, bias, args, layer_name):

        # Qinput, Iscale, Qinput_t = fp8_division_transpose(input, 16, args.fabit)
        Qinput, Iscale, Qinput_t = fp8_quantize_pertensor_transpose(input, 16, args.fabit, transpose_output_2d=True)

        # Qweight, Wscale, Qweight_t = fp8_division_transpose(weight, 16, args.fwbit)
        Qweight, Wscale, Qweight_t = fp8_quantize_pertensor_transpose(weight, 16, args.fwbit, transpose_output_2d=True)

        ctx.saved = Qinput_t, Iscale, Qweight_t, Wscale, bias, args, layer_name
        fc_output = fp8_linear_forward(Qinput, Iscale, Qweight, Wscale, False, 0, bias)

        return fc_output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):

        Qinput_t, Iscale, Qweight_t, Wscale, bias, args, layer_name = ctx.saved

        # Qgrad_output, Gscale, Qgrad_output_t = fp8_division_transpose(grad_output, 16, args.bobit, stochastic=False)
        Qgrad_output, Gscale, Qgrad_output_t = fp8_quantize_pertensor_transpose(
            grad_output, 16, args.bobit, stochastic=False, transpose_output_2d=True
        )

        grad_input, grad_weight = fp8_linear_backward(
            Qinput_t,
            Iscale,
            Qgrad_output,
            Gscale,
            Qgrad_output_t,
            Qweight_t,
            Wscale,
            16,
            bias,
            stochastic=False,
            dgrad_quantize=False,
        )

        if bias is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None



# if int(os.environ.get("LOCAL_RANK")) == 0:
#     import IPython
#     IPython.embed()
# else:
#     import time
#     time.sleep(1000)

# class QuantLinearTE(Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, args, layer_type):
#         ctx.saved = input, weight, bias, args, layer_type
#         return F.linear(input, weight, bias)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight, bias, args, layer_type = ctx.saved

#         C_in = input.shape[-1]
#         C_out = grad_output.shape[-1]

#         grad_output_flatten = grad_output.reshape(-1, C_out)
#         input_flatten = input.reshape(-1, C_in)

#         grad_weight = grad_output_flatten.t().mm(input_flatten)
#         grad_input = grad_output_flatten.mm(weight)
        
#         if bias is not None:
#             grad_bias = grad_output_flatten.sum(0)
#         else:
#             grad_bias = None

#         grad_input_transform = grad_input.reshape(input.size())

#         return grad_input_transform, grad_weight, grad_bias, None, None

