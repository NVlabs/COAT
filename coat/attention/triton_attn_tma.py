"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

# Copied from commit: https://github.com/triton-lang/triton/commit/3d1aeca3f2788d7509beea08a9c1a7e7c8ac89b7
"""

import pytest
import torch
import triton.tools.experimental_descriptor

import triton
import triton.language as tl
try:
    from .tma_utils import TmaAutoTuneHelper, supports_tma
except:
    from tma_utils import TmaAutoTuneHelper, supports_tma

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                        desc_k, desc_v,  #
                        offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                        N_CTX: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl._experimental_descriptor_load(desc_k, [offsetkv_y, 0], [BLOCK_N, HEAD_DIM], dtype).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl._experimental_descriptor_load(desc_v, [offsetkv_y, 0], [BLOCK_N, HEAD_DIM], dtype)
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        offsetkv_y += BLOCK_N
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs_tma = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in [2, 3, 4, 6]\
    for w in [4, 8]\
]


def keep_tma(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if (torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8):
        return False
    return True


@triton.autotune(configs=list(filter(keep_tma, configs_tma)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
@triton.jit
def _attn_fwd_tma(sm_scale, M,  #
                  Z, H, desc_q, desc_k, desc_v, desc_o,  #
                  q, # This is only used for dtype
                  N_CTX,  #
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  FP8_OUTPUT: tl.constexpr,  #
                  STAGE: tl.constexpr  #
                  ):
    dtype = q.dtype.element_ty
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl._experimental_descriptor_load(desc_q, [qo_offset_y, 0], [BLOCK_M, HEAD_DIM], dtype)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                                            desc_k, desc_v,  #
                                            offset_y, dtype, start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            4 - STAGE, offs_m, offs_n, N_CTX,  #
                                            )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                                            desc_k, desc_v,  #
                                            offset_y, dtype, start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX,  #
                                            )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl._experimental_descriptor_store(desc_o, acc.to(dtype), [qo_offset_y, 0])


def tma_attention_forward(q, k, v, causal, sm_scale):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    extra_kern_args = {}

    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("q")
    desc_helper.init_tma_descriptor("v")
    desc_helper.init_tma_descriptor("k")
    desc_helper.init_tma_descriptor("o")

    def grid(META):
        nonlocal desc_helper

        desc_helper.fill_2d_tma_descriptor("q", q.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_M"], HEAD_DIM_K,
                                            q.element_size())

        desc_helper.fill_2d_tma_descriptor("v", v.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_N"], HEAD_DIM_K,
                                            v.element_size())

        desc_helper.fill_2d_tma_descriptor("k", k.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_N"], HEAD_DIM_K,
                                            k.element_size())

        desc_helper.fill_2d_tma_descriptor("o", o.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_M"], HEAD_DIM_K,
                                            o.element_size())

        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    desc_q = desc_helper.get_tma_descriptor_kernel_param("q")
    desc_v = desc_helper.get_tma_descriptor_kernel_param("v")
    desc_k = desc_helper.get_tma_descriptor_kernel_param("k")
    desc_o = desc_helper.get_tma_descriptor_kernel_param("o")

    _attn_fwd_tma[grid](
        sm_scale, M,  #
        q.shape[0], q.shape[1],  #
        desc_q, desc_k, desc_v, desc_o,
        q, # This is only used for dtype
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        FP8_OUTPUT=q.dtype in [torch.float8_e5m2, torch.float8_e4m3fn],  #
        STAGE=stage,  #
        **extra_kern_args)
    return o
