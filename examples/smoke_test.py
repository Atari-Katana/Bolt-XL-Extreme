
import torch
import logging
from typing import Optional

# Configure robust logging for the swarm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntegrationTeam")

def smoke_test_inference():
    logger.info("PHASE 4: Initiating Smoke Test")
    
    try:
        import vllm
        from vllm import LLM, SamplingParams
        logger.info(f"✅ Module 'vllm' loaded successfully. Version: {vllm.__version__}")
    except ImportError as e:
        logger.error(f"❌ Failed to load vllm module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error loading vllm: {e}")
        return False

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! Verification failed.")
        return False
    
    logger.info(f"✅ CUDA Available. Device: {torch.cuda.get_device_name(0)}")

    try:
        # Mini-test with a small model (using OPT-125m or similar small local model if available, 
        # but defaulting to a standard dummy verification pattern if appropriate)
        # Note: In a real environment we might download 'facebook/opt-125m', 
        # but for smoke testing we want to minimize network dependency if possible. 
        # We'll assume internet access is available or use a mocked approach.
        
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
        ]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)

        logger.info("Initializing LLM Engine (Mock/Small Model)...")
        # Using a small, safe model for smoke testing with constrained resources
        llm = LLM(
            model="facebook/opt-125m", 
            dtype="half", 
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            max_model_len=512,
            enforce_eager=True # Avoid CUDAGraphs overhead for smoke test
        ) 
        
        logger.info("Running Inference...")
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            logger.info(f"Generated: {prompt!r} -> {generated_text!r}")
            
            # Validation: Ensure we got *some* non-empty text
            if not generated_text:
                logger.error("❌ Inference generated empty output!")
                return False

        logger.info("✅ Inference path verified. Shapes/Tensors appear valid.")
        return True

    except Exception as e:
        logger.error(f"❌ Smoke Test Failed during Inference: {e}")
        return False

if __name__ == "__main__":
    success = smoke_test_inference()
    if success:
        logger.info("🚀 PHASE 4 COMPLETE: SMOKE TEST PASSED")
        exit(0)
    else:
        logger.error("💀 PHASE 4 FAILED: SMOKE TEST FAILURE")
        exit(1)
