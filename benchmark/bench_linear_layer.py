import torch
import torch.nn as nn
import triton
from coat.activation.real_quantization.linear.fp8linear import FP8Linear
from coat.activation.deepseek.fp8linear import FP8DeepSeekLinear

torch.cuda.empty_cache()
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['k', 'm'],  # argument names to use as an x-axis for the plot
        x_vals=[1024 * i for i in range(1, 17)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['bf16', 'fp8-per-tensor', 'fp8-deepseek'],  # possible values for `line_arg``
        line_names=[
            "bf16",
            "fp8-per-tensor",
            "fp8-deepseek",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="backward",  # name for the plot. Used also as a file name for saving the plot.
        args={},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(m, k, provider):
    n = k
    device = 'cuda'
    dtype = torch.bfloat16
    x = torch.randn(m, k, dtype=dtype, device=device) 
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'bf16':
        fc = torch.nn.Linear(k,n, bias=False, dtype=dtype, device=device)
        ms = triton.testing.do_bench(lambda: fc(x), rep=500)
    if provider == 'fp8-per-tensor':
        fc = FP8Linear(k, n, bias=False, device=device).to(torch.bfloat16)
        ms = triton.testing.do_bench(lambda: fc(x), rep=500)
    if provider == 'fp8-deepseek':
        fc = FP8DeepSeekLinear(k, n, bias=False, device=device).to(torch.bfloat16)
        ms = triton.testing.do_bench(lambda: fc(x), rep=500)
    return ms
benchmark.run(show_plots=True, print_data=True)

# =============================== Backward ===============================

torch.cuda.empty_cache()
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['k', 'm'],  # argument names to use as an x-axis for the plot
        x_vals=[1024 * i for i in range(1, 17)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['bf16', 'fp8-per-tensor', 'fp8-deepseek'],  # possible values for `line_arg``
        line_names=[
            "bf16",
            "fp8-per-tensor",
            "fp8-deepseek",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="backward",  # name for the plot. Used also as a file name for saving the plot.
        args={},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(m, k, provider):
    n = k
    device = 'cuda'
    dtype = torch.bfloat16
    x = torch.randn(m, k, dtype=dtype, device=device)
    x.requires_grad_(True)
    
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'bf16':
        fc = torch.nn.Linear(k,n, bias=False, dtype=dtype, device=device)
        y = fc(x)
        dy = torch.rand_like(y)
        ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), grad_to_none=[x])
    if provider == 'fp8-per-tensor':
        fc = FP8Linear(k,n, bias=False, device=device).to(torch.bfloat16)
        y = fc(x)
        dy = torch.rand_like(y)
        ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), grad_to_none=[x])
    if provider == 'fp8-deepseek':
        fc = FP8DeepSeekLinear(k,n, bias=False, device=device).to(torch.bfloat16)
        y = fc(x)
        dy = torch.rand_like(y)
        ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), grad_to_none=[x])

    return ms
benchmark.run(show_plots=True, print_data=True)