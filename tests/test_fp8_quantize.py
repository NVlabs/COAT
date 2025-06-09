import unittest

import torch
from .test_utils import (check_similarity, dequantize_tensor, quantize_tensor,
                        random_tensor_generator)

from coat.activation.real_quantization.common import FP8_MAX_VALUE
from coat.activation.real_quantization.quantize._quantize_perblock import fp8_quantize_perblock
from coat.activation.real_quantization.quantize._quantize_transpose import fp8_quantize_transpose


def _test_quantize_perblock(x, BS, SL, CDIM, fp8type, QB, eps=1e-5):
    """
    Test the forward and backward of LayerNorm (no param)
    Forward: input is per-group quant, output is per-tensor quanti
    Backward: input and output are full precision
    """
    x = x.clone().requires_grad_(True)
    output_x, output_qx, output_sx = quantize_tensor(x, BS, SL, CDIM, QB, fp8type, quant_type="per_block")
    
    triton_qx, triton_sx = fp8_quantize_perblock(x, QB, fp8type)
    return (
        output_qx,
        output_sx,
        triton_qx,
        triton_sx,
    )


class TestNorm(unittest.TestCase):
    """
    x: reconstructed activation (quantize it then dequantize)
    qx: quantized activation
    sx: scale factor of activation

    g: reconstrcted gradient (quantize it then dequantize)
    qg: quantized gradient
    sg: scale factor of gradient
    """

    # Test the forward of LayerNorm
    def test_quantize_perblock(self):
        BS, SL, CDIM, QB, fp8type = [4, 256, 2048, 16, torch.float8_e4m3fn]
        x = torch.randn(BS * SL, CDIM).cuda()
        output_qx, output_sx, triton_qx, triton_sx = _test_quantize_perblock(x, BS, SL, CDIM, fp8type, QB)

        self.assertTrue(check_similarity(output_qx, triton_qx))
        self.assertTrue(check_similarity(output_sx, triton_sx))

    def test_quantize_transpose(self):
        BS, SL, CDIM, QB, fp8type = [16, 8192, 11008, 128, torch.float8_e4m3fn]
        x = torch.randn(BS * SL, CDIM, dtype=torch.bfloat16, device="cuda:1")
        
        torch.cuda.synchronize()
        
        with torch.cuda.device(x.device):
            fp8_quantize_transpose(x, QB, fp8type, scale_dtype=torch.float32)
        
        
if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200)
    unittest.main()
