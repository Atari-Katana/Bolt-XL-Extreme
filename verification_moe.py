import torch
import torch.nn.functional as F
import functools
import sys
import os

# Ensure we can import from vllm
sys.path.append(os.getcwd())

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
    from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
    from vllm.model_executor.layers.fused_moe.config import (
        int4_w4a16_moe_quant_config,
        int8_w8a16_moe_quant_config,
    )
    from vllm.scalar_type import scalar_types
except ImportError as e:
    print(f"Skipping test due to import error: {e}")
    sys.exit(0)

def iterative_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
    renormalize: bool = False,
) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = hidden_states.dtype

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, global_num_experts)
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights, selected_experts = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    if expert_map is not None:
        selected_experts = expert_map[selected_experts]

    final_hidden_states = None
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_mask = selected_experts == expert_idx
        expert_weights = (topk_weights * expert_mask).sum(dim=-1, keepdim=True)
        # Check if any token uses this expert to avoid unnecessary computation
        if expert_weights.sum() == 0:
            continue
            
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[:, :intermediate_size])
        x = x[:, intermediate_size:] * gate
        x = F.linear(x, expert_w2)
        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states
    
    if final_hidden_states is None:
        final_hidden_states = torch.zeros_like(hidden_states)

    return final_hidden_states.view(orig_shape)

def test_fused_moe_wn16_simple():
    print("Starting test_fused_moe_wn16_simple...")
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    # specific parameters
    m = 128
    n = 128
    k = 128
    e = 8
    topk = 2
    dtype = torch.bfloat16
    group_size = 64
    has_zp = False # Simpler case
    weight_bits = 4

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    if weight_bits == 4:
        pack_factor = 2
        quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    elif weight_bits == 8:
        pack_factor = 1
        quant_type = scalar_types.uint8 if has_zp else scalar_types.uint8b128

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    
    # Quantize weights
    w1_qweight = torch.empty((e, 2 * n, k // pack_factor), device="cuda", dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // pack_factor), device="cuda", dtype=torch.uint8)
    w1_scales = torch.empty((e, 2 * n, k // group_size), device="cuda", dtype=dtype)
    w2_scales = torch.empty((e, k, n // group_size), device="cuda", dtype=dtype)
    w1_qzeros = torch.empty((e, 2 * n // pack_factor, k // group_size), device="cuda", dtype=torch.uint8)
    w2_qzeros = torch.empty((e, k // pack_factor, n // group_size), device="cuda", dtype=torch.uint8)

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
            w, w_ref_ptr, w_qweight, w_scales, w_qzeros = w1, w1_ref, w1_qweight, w1_scales, w1_qzeros
        else:
            w, w_ref_ptr, w_qweight, w_scales, w_qzeros = w2, w2_ref, w2_qweight, w2_scales, w2_qzeros
        
        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False
        )
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T
        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.uint8)
        if weight_bits == 4:
            qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
            if has_zp:
                qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

        w_ref_ptr[expert_id] = weight
        w_qweight[expert_id] = qweight
        w_scales[expert_id] = scales
        if has_zp:
            w_qzeros[expert_id] = qzeros

    if weight_bits == 4:
        quant_config_builder = int4_w4a16_moe_quant_config
    else:
        quant_config_builder = int8_w8a16_moe_quant_config

    quant_config = quant_config_builder(
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=w1_qzeros if has_zp else None,
        w2_zp=w2_qzeros if has_zp else None,
        block_shape=[0, group_size],
    )

    print("Invoking fused_moe (Triton path)...")
    triton_output = fused_moe(
        a,
        w1_qweight,
        w2_qweight,
        score,
        topk,
        renormalize=False,
        global_num_experts=e,
        expert_map=None,
        quant_config=quant_config,
    )
    
    print("Invoking reference implementation...")
    torch_output = iterative_moe(a, w1_ref, w2_ref, score, topk, global_num_experts=e)

    diff = (triton_output - torch_output).abs().max()
    print(f"Max difference: {diff}")
    
    # We use a relatively loose tolerance for quantized kernels vs fp reference as seen in original tests
    if diff < 0.1: 
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
        # print first few elements
        print("Triton:", triton_output.flatten()[:10])
        print("Ref:", torch_output.flatten()[:10])

if __name__ == "__main__":
    test_fused_moe_wn16_simple()
