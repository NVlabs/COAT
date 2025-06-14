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
from olmo.config import CheckpointType, TrainConfig
from olmo.model import BufferCache, LayerNorm, OLMoSequentialBlock

from coat.utils._fp8manager import FP8Manager
from coat.models.coat_olmo import CoatOLMoSequentialBlock
from coat.activation.real_quantization import (Coat_quantize_bgn,
                                               Coat_quantize_end)

torch.manual_seed(0)
batch_sizes = [8]
sequence_lengths = [2048]
n_repeat = 20

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # Triton will throw RuntimeError: Triton Error [CUDA]: invalid device context if you comment this line


def benchmarker(CHOICE, OBJECT, verbose=False):
    if CHOICE == "BF16":
        yaml_path = "examples/OLMo/configs/reproduce/OLMo-7B-reproduce.yaml"
        # yaml_path = "examples/OLMo/configs/reproduce/OLMo-1B-reproduce.yaml"
        cfg = TrainConfig.load(yaml_path)
        cfg.model.init_device = "cuda"
        olmo_layer = OLMoSequentialBlock(0, cfg.model, BufferCache())
        olmo_layer = olmo_layer.cuda().to(torch.bfloat16)

        model_memory = torch.cuda.memory_allocated()
        if verbose:
            print(f"Model Memory {model_memory / 1024 ** 2} MB")

        # Forward Time Benchmark
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                dummy_input = (
                    torch.rand((batch_size, sequence_length, cfg.model.d_model), dtype=torch.bfloat16)
                    .cuda()
                    .requires_grad_(True)
                )
                dummy_grad = torch.rand((batch_size, sequence_length, cfg.model.d_model), dtype=torch.bfloat16).cuda()

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
                            dummy_output = olmo_layer(dummy_input)
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
                            dummy_output = olmo_layer(dummy_input)
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
                            dummy_output = olmo_layer(dummy_input)
                        end_memory = torch.cuda.memory_allocated()

                        dummy_output[0].backward(dummy_grad)

                        memory_of_output = dummy_output[0].element_size() * dummy_output[0].numel()
                        end_memory = end_memory - memory_of_output
                        del dummy_output
                        dummy_input.grad = None
                        for n, p in olmo_layer.named_parameters():
                            p.grad = None
                        torch.cuda.empty_cache()

                        if verbose:
                            print(
                                f"Batch Size: {batch_size} | Sequence_length: {sequence_length} | Activation Memory: {(end_memory - start_memory) / 1024 ** 2} MB | "
                                f"Start Memory {start_memory / 1024 ** 2} MB | End Memory {end_memory / 1024 ** 2} MB"
                            )

                    return (end_memory - start_memory) / 1024**2  # MB

    if CHOICE == "COAT":
        yaml_path = "examples/OLMo/configs/coat/OLMo-7B-COAT-Activation.yaml"
        # yaml_path = "examples/OLMo/configs/coat/OLMo-1B-realquantize-model-g16_pertensor.yaml"
        cfg = TrainConfig.load(yaml_path)
        cfg.model.init_device = "cuda"
        if cfg.quantize_model.refine_row_blocksize == 1 and cfg.quantize_model.refine_col_blocksize > 1:
            cfg.quantize_model.group_size = cfg.quantize_model.refine_col_blocksize
        olmo_layer = CoatOLMoSequentialBlock(0, cfg.model, cfg.quantize_model, BufferCache())
        olmo_bgn = Coat_quantize_bgn(cfg.quantize_model)
        olmo_end = Coat_quantize_end(cfg.quantize_model)
        olmo_layer = olmo_layer.cuda().to(torch.bfloat16)

        # Forward Time Benchmark
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                dummy_input = (
                    torch.rand((batch_size, sequence_length, cfg.model.d_model), dtype=torch.bfloat16)
                    .cuda()
                    .requires_grad_(True)
                )
                dummy_grad = torch.rand((batch_size, sequence_length, cfg.model.d_model), dtype=torch.bfloat16).cuda()

                tensor_memory = torch.cuda.memory_allocated()
                if verbose:
                    print(f"Tensor Memory {tensor_memory / 1024 ** 2} MB")

                start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
                end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

                if OBJECT == "SPEED":
                    FP8Manager.is_first_microbatch = True
                    dummy_x, dummy_qx, dummy_sx = olmo_bgn(dummy_input)
                    _ = olmo_layer(dummy_x, dummy_qx, dummy_sx)

                    # Test Forward Time
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        FP8Manager.is_first_microbatch = False
                        for i in range(n_repeat):
                            start_event[i].record()
                            dummy_output = olmo_layer(dummy_x, dummy_qx, dummy_sx)
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
                            dummy_output_x, dummy_output_qx, dummy_output_sx, dummy_output_cache = olmo_layer(
                                dummy_x, dummy_qx, dummy_sx
                            )
                            dummy_output = olmo_end(dummy_output_x, dummy_output_qx, dummy_output_sx)
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
                            dummy_x, dummy_qx, dummy_sx = olmo_bgn(dummy_input)

                            start_memory = torch.cuda.memory_allocated()
                            dummy_output_x, dummy_output_qx, dummy_output_sx, dummy_output_cache = olmo_layer(
                                dummy_x, dummy_qx, dummy_sx
                            )
                            end_memory = torch.cuda.memory_allocated()
                            dummy_output = olmo_end(dummy_output_x, dummy_output_qx, dummy_output_sx)

                        # end_memory = torch.cuda.memory_allocated()
                        dummy_output.backward(dummy_grad)

                        memory_of_output = sum(
                            [t.element_size() * t.numel() for t in [dummy_output_qx, dummy_output_sx]]
                        )
                        end_memory = end_memory - memory_of_output
                        del dummy_output, dummy_qx, dummy_sx, dummy_output_qx, dummy_output_sx
                        dummy_input.grad = None
                        for n, p in olmo_layer.named_parameters():
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
coat_memory_consumption = benchmarker("COAT", "MEMORY", verbose=False)
bf16_memory_consumption = benchmarker("BF16", "MEMORY", verbose=False)

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
