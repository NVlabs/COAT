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

# This repo is used to reproduce Table 6 in COAT paper

import pandas as pd
import torch
from dataclasses import asdict
from transformers import AutoConfig, AutoModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from coat.utils._fp8manager import FP8Manager
from coat.utils._fp8_quantization_config import QuantizationConfig
from coat.models.coat_llama import CoatLlamaDecoderLayer, CoatLlamaModel, make_state_dict_compatible
from coat.activation.real_quantization import (Coat_quantize_bgn,
                                               Coat_quantize_end)

torch.manual_seed(0)
batch_sizes = [8]
sequence_lengths = [2048]
n_repeat = 20

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # Triton will throw RuntimeError: Triton Error [CUDA]: invalid device context if you comment this line

# model_name = "meta-llama/Llama-3.1-8B"
model_name = "meta-llama/Llama-2-7b-hf"
config = AutoConfig.from_pretrained(model_name)
config._attn_implementation = "flash_attention_2"
model = AutoModel.from_pretrained(model_name)

def benchmarker(CHOICE, OBJECT, verbose=False):
    if CHOICE == "BF16":
        llama_layer = LlamaDecoderLayer(config=config, layer_idx=0)
        llama_layer = llama_layer.cuda().to(torch.bfloat16)
        llama_layer.load_state_dict(model.layers[0].state_dict())

        model_memory = torch.cuda.memory_allocated()
        if verbose:
            print(f"Model Memory {model_memory / 1024 ** 2} MB")

        # Forward Time Benchmark
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                dummy_input = (
                    torch.rand((batch_size, sequence_length, config.hidden_size), dtype=torch.bfloat16)
                    .cuda()
                    .requires_grad_(True)
                )

                attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool).cuda()
                position_ids = torch.arange(sequence_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(batch_size, -1)

                dummy_grad = torch.rand((batch_size, sequence_length, config.hidden_size), dtype=torch.bfloat16).cuda()

                tensor_memory = torch.cuda.memory_allocated()
                if verbose:
                    print(f"Tensor Memory {tensor_memory / 1024 ** 2} MB")

                if OBJECT == "SPEED":
                    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
                    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

                    FP8Manager.is_first_microbatch = True

                    # Test Forward Time
                    with torch.autocast(device_type="cuda:7", dtype=torch.bfloat16):
                        FP8Manager.is_first_microbatch = False
                        for i in range(n_repeat):
                            start_event[i].record()
                            dummy_output = llama_layer(dummy_input, attention_mask, position_ids)
                            end_event[i].record()
                        torch.cuda.synchronize()

                    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
                    avg_time_fwd = torch.median(times)

                    if verbose:
                        print(times)
                        print(
                            f"Forward - Batch Size: {batch_size} | Sequence_length: {sequence_length} | Time: {avg_time_fwd}"
                        )

                    # Test Backward Time
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        torch.empty(
                            1, device="cuda", requires_grad=True
                        ).backward()  # Triton will throw RuntimeError: Triton Error [CUDA]: invalid device context if you comment this line
                        for i in range(n_repeat):
                            dummy_output = llama_layer(dummy_input, attention_mask, position_ids)
                            start_event[i].record()
                            dummy_output[0].backward(dummy_grad)
                            end_event[i].record()
                        torch.cuda.synchronize()

                    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
                    avg_time_bwd = torch.median(times)

                    if verbose:
                        print(times)
                        print(
                            f"Backward - Batch Size: {batch_size} | Sequence_length: {sequence_length} | Time: {avg_time_bwd}"
                        )

                    return avg_time_fwd, avg_time_bwd, avg_time_fwd + avg_time_bwd

                if OBJECT == "MEMORY":
                    # Test Memory Consumption
                    for _ in range(n_repeat):
                        torch.cuda.empty_cache()
                        start_memory = torch.cuda.memory_allocated()
                        # with te.fp8_autocast(enabled=True):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            dummy_output = llama_layer(dummy_input, attention_mask, position_ids)
                        end_memory = torch.cuda.memory_allocated()

                        dummy_output[0].backward(dummy_grad)

                        memory_of_output = dummy_output[0].element_size() * dummy_output[0].numel()
                        end_memory = end_memory - memory_of_output
                        del dummy_output
                        dummy_input.grad = None
                        for n, p in llama_layer.named_parameters():
                            p.grad = None
                        torch.cuda.empty_cache()

                        if verbose:
                            print(
                                f"Batch Size: {batch_size} | Sequence_length: {sequence_length} | Activation Memory: {(end_memory - start_memory) / 1024 ** 2} MB | "
                                f"Start Memory {start_memory / 1024 ** 2} MB | End Memory {end_memory / 1024 ** 2} MB"
                            )

                    return (end_memory - start_memory) / 1024**2  # MB

    if CHOICE == "COAT":
        fp8_config = QuantizationConfig(
            quantize_model=True,
            group_size=16,
        )
        config.coat_fp8_args = asdict(fp8_config)

        llama_layer = CoatLlamaDecoderLayer(config, layer_idx=0)
        llama_layer.load_state_dict(make_state_dict_compatible(model.layers[0].state_dict()))

        llama_bgn = Coat_quantize_bgn(fp8_config).cuda()
        llama_end = Coat_quantize_end(fp8_config).cuda()
        llama_layer = llama_layer.cuda().to(torch.bfloat16)

        # Forward Time Benchmark
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                dummy_input = (
                    torch.rand((batch_size, sequence_length, config.hidden_size), dtype=torch.bfloat16)
                    .cuda()
                    .requires_grad_(True)
                )
                
                attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool).cuda()
                position_ids = torch.arange(sequence_length, dtype=torch.long, device="cuda").unsqueeze(0).expand(batch_size, -1)

                dummy_grad = torch.rand((batch_size, sequence_length, config.hidden_size), dtype=torch.bfloat16).cuda()

                tensor_memory = torch.cuda.memory_allocated()
                if verbose:
                    print(f"Tensor Memory {tensor_memory / 1024 ** 2} MB")

                start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
                end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

                if OBJECT == "SPEED":
                    FP8Manager.is_first_microbatch = True
                    dummy_x, dummy_qx, dummy_sx = llama_bgn(dummy_input)
                    _ = llama_layer(dummy_x, dummy_qx, dummy_sx, attention_mask, position_ids)[0]

                    # Test Forward Time
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        FP8Manager.is_first_microbatch = False
                        for i in range(n_repeat):
                            start_event[i].record()
                            dummy_output = llama_layer(dummy_x, dummy_qx, dummy_sx, attention_mask, position_ids)
                            end_event[i].record()
                        torch.cuda.synchronize()

                    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
                    avg_time_fwd = torch.median(times)

                    if verbose:
                        print(times)
                        print(
                            f"Forward - Batch Size: {batch_size} | Sequence_length: {sequence_length} | Time: {avg_time_fwd}"
                        )

                    # Test Backward Time
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        torch.empty(
                            1, device="cuda", requires_grad=True
                        ).backward()  # Triton will throw RuntimeError: Triton Error [CUDA]: invalid device context if you comment this line
                        for i in range(n_repeat):
                            dummy_output_x, dummy_output_qx, dummy_output_sx = llama_layer(
                                dummy_x, dummy_qx, dummy_sx, attention_mask, position_ids
                            )[0]
                            dummy_output = llama_end(dummy_output_x, dummy_output_qx, dummy_output_sx)
                            start_event[i].record()
                            dummy_output.backward(dummy_grad)
                            end_event[i].record()
                        torch.cuda.synchronize()

                    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
                    avg_time_bwd = torch.median(times)

                    if verbose:
                        print(times)
                        print(
                            f"Backward - Batch Size: {batch_size} | Sequence_length: {sequence_length} | Time: {avg_time_bwd}"
                        )

                    return avg_time_fwd, avg_time_bwd, avg_time_fwd + avg_time_bwd

                if OBJECT == "MEMORY":
                    # Test Memory Consumption
                    for _ in range(n_repeat):
                        torch.cuda.empty_cache()
                        # start_memory = torch.cuda.memory_allocated()
                        # with te.fp8_autocast(enabled=True):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            if _ == 0:
                                FP8Manager.is_first_microbatch = True
                            else:
                                FP8Manager.is_first_microbatch = False
                            dummy_x, dummy_qx, dummy_sx = llama_bgn(dummy_input)

                            start_memory = torch.cuda.memory_allocated()
                            dummy_output_x, dummy_output_qx, dummy_output_sx = llama_layer(
                                dummy_x, dummy_qx, dummy_sx, attention_mask, position_ids
                            )[0]
                            end_memory = torch.cuda.memory_allocated()
                            dummy_output = llama_end(dummy_output_x, dummy_output_qx, dummy_output_sx)

                        # end_memory = torch.cuda.memory_allocated()
                        dummy_output.backward(dummy_grad)

                        memory_of_output = sum(
                            [t.element_size() * t.numel() for t in [dummy_output_qx, dummy_output_sx]]
                        )
                        end_memory = end_memory - memory_of_output
                        del dummy_output, dummy_qx, dummy_sx, dummy_output_qx, dummy_output_sx
                        dummy_input.grad = None
                        for n, p in llama_layer.named_parameters():
                            p.grad = None
                        torch.cuda.empty_cache()

                        if verbose:
                            print(
                                f"Batch Size: {batch_size} | Sequence_length: {sequence_length} | Activation Memory: {(end_memory - start_memory) / 1024 ** 2} MB | "
                                f"Start Memory {start_memory / 1024 ** 2} MB | End Memory {end_memory / 1024 ** 2} MB"
                            )

                    return (end_memory - start_memory) / 1024**2  # MB


# Benchmark the memory reduction of COAT
print("========== Benchmarking the Memory Reduction ability of COAT ==========")
coat_memory_consumption = benchmarker("COAT", "MEMORY", verbose=True)
bf16_memory_consumption = benchmarker("BF16", "MEMORY", verbose=True)

print(
    f"BF16 consumes: {bf16_memory_consumption} MB per layer\n"
    f"COAT consumes: {coat_memory_consumption} MB per layer\n"
    f"We achieve {(bf16_memory_consumption / coat_memory_consumption):.2f} times memory footprint reduction\n"
)

# Benchmark the speedup of COAT
print("========== Benchmarking the Speedup ability of COAT ==========")
coat_avg_time_fwd, coat_avg_time_bwd, coat_avg_time_total = benchmarker("COAT", "SPEED", verbose=False)
bf16_avg_time_fwd, bf16_avg_time_bwd, bf16_avg_time_total = benchmarker("BF16", "SPEED", verbose=False)

print(
    f"BF16 consumes: {bf16_avg_time_total}ms per layer. {bf16_avg_time_fwd}ms in forward | {bf16_avg_time_bwd}ms in backward\n"
    f"COAT consumes: {coat_avg_time_total}ms per layer. {coat_avg_time_fwd}ms in forward | {coat_avg_time_bwd}ms in backward\n"
    f"We achieve {(bf16_avg_time_total / coat_avg_time_total):.2f} times speedup per layer"
)
