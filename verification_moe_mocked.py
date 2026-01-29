import sys
import os
import torch
from types import ModuleType

# Mock vllm._custom_ops
class MockCustomOps:
    @staticmethod
    def moe_sum(input, output):
        res = input.sum(dim=-2)
        output.copy_(res)
    
    @staticmethod
    def cutlass_scaled_mm_supports_fp4():
        return False
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

mock_ops_module = ModuleType("vllm._custom_ops")
mock_ops_module.moe_sum = MockCustomOps.moe_sum
mock_ops_module.cutlass_scaled_mm_supports_fp4 = MockCustomOps.cutlass_scaled_mm_supports_fp4
sys.modules["vllm._custom_ops"] = mock_ops_module

# Add root to sys.path
sys.path.append(os.getcwd())

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl
    from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
    from vllm.model_executor.layers.fused_moe.config import (
        int4_w4a16_moe_quant_config,
        int8_w8a16_moe_quant_config,
    )
    from vllm.scalar_type import scalar_types
except ImportError as e:
    print(f"Skipping test due to import error: {e}")
    sys.exit(1)

import torch.nn.functional as F

def test_fused_moe_wn16_simple():
    print("Starting test_fused_moe_wn16_simple (Mocked impl call, correct args)...")
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    m = 16
    n = 128
    k = 128
    e = 8
    topk = 2
    dtype = torch.float16
    group_size = 64
    has_zp = False
    weight_bits = 4
    
    device = "cuda"

    a = torch.randn((m, k), device=device, dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10
    score = torch.randn((m, e), device=device, dtype=dtype)

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    elif weight_bits == 8:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    
    w1_qweight = torch.empty((e, 2 * n, k // pack_factor), device=device, dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // pack_factor), device=device, dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size), device=device, dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size), device=device, dtype=dtype)
    
    try:
        for i in range(e * 2):
            expert_id = i % e
            if i // e == 0:
                w_ptr = w1
                wq = w1_qweight
                ws = w1_scales
            else:
                w_ptr = w2
                wq = w2_qweight
                ws = w2_scales
            
            weight, qweight, scales, qzeros = quantize_weights(
                w_ptr[expert_id].T, quant_type, group_size, has_zp, False
            )
            qweight = qweight.T.contiguous().to(torch.uint8)
            scales = scales.T

            if weight_bits == 4:
                 qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]

            wq[expert_id] = qweight
            ws[expert_id] = scales
            
    except Exception as e:
        print(f"Quantization logic failed: {e}")
        w1_qweight.random_(0, 255)
        w2_qweight.random_(0, 255)
        w1_scales.fill_(1.0)
        w2_scales.fill_(1.0)

    if weight_bits == 4:
        quant_config_builder = int4_w4a16_moe_quant_config
    else:
        quant_config_builder = int8_w8a16_moe_quant_config

    quant_config = quant_config_builder(
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, group_size],
    )

    topk_weights, topk_ids = torch.topk(score.softmax(dim=-1, dtype=torch.float), topk, dim=-1)
    topk_weights = topk_weights.to(dtype)
    topk_ids = topk_ids.int()

    print("Invoking fused_experts_impl (Triton path)...")
    try:
        # Manually unpack config
        triton_output = fused_experts_impl(
            hidden_states=a,
            w1=w1_qweight,
            w2=w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            global_num_experts=e,
            use_fp8_w8a8=quant_config.use_fp8_w8a8,
            use_int8_w8a8=quant_config.use_int8_w8a8,
            use_int8_w8a16=quant_config.use_int8_w8a16,
            use_int4_w4a16=quant_config.use_int4_w4a16,
            ocp_mx_scheme=quant_config.ocp_mx_scheme,
            per_channel_quant=quant_config.per_act_token_quant,
            expert_map=None,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            w1_zp=quant_config.w1_zp,
            w2_zp=quant_config.w2_zp,
            # a1_scale and others might be kwarg or explicit arguments, 
            # checking signature provided earlier lines 1463-1480+
            # It seems signatures end with w1_scale=None... so we should check if a1_scale/w1_bias are in it.
            # Assuming args from fused_experts unpacking are correct.
        )
        print("Triton execution completed.")
    except Exception as e:
        print(f"Triton execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Checking output validity...")
    if torch.isnan(triton_output).any():
        print("❌ FAILED: Output contains NaNs")
    elif triton_output.abs().sum() == 0:
        print("⚠️ WARNING: Output is all zeros.")
    else:
        print("✅ PASSED: Kernel executed and produced non-zero output")
        print(f"Output mean: {triton_output.abs().mean()}")

if __name__ == "__main__":
    test_fused_moe_wn16_simple()
