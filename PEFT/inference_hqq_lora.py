#!/usr/bin/env python3
"""
Inference script for HQQ quantized model with LoRA adapters
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from hqq.models.hf.base import AutoHQQHFModel
from hqq.models.base import HQQLinear
from hqq.core.quantize import BaseQuantizeConfig, HQQBackend
from hqq.utils.patching import prepare_for_inference, recommended_inductor_config_setter
import time
import logging


def setup_logging():
    """Setup logging"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def load_model_and_adapters(model_path, adapter_path, device="cuda"):
    """Load HQQ quantized model with LoRA adapters"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading base model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load LoRA adapters if specified
    if adapter_path:
        logger.info(f"Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge adapters for inference
    
    model.eval()
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generate text using the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        start_time = time.time()
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        end_time = time.time()
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Calculate metrics
    generation_time = end_time - start_time
    tokens_generated = outputs[0].shape[1] - inputs.input_ids.shape[1]
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    return response, tokens_per_second, generation_time


def interactive_mode(model, tokenizer):
    """Interactive mode for testing the model"""
    print("Interactive mode started. Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        prompt = input("\nEnter your prompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt.strip():
            continue
        
        print("Generating response...")
        response, tps, gen_time = generate_text(model, tokenizer, prompt)
        
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
        print(f"Generation time: {gen_time:.2f}s")
        print(f"Tokens per second: {tps:.2f}")
        print("-" * 50)


def benchmark_mode(model, tokenizer, prompts, max_new_tokens=100):
    """Benchmark mode for testing performance"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running benchmark with {len(prompts)} prompts...")
    
    total_time = 0
    total_tokens = 0
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}")
        
        response, tps, gen_time = generate_text(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        
        total_time += gen_time
        total_tokens += len(tokenizer.encode(response))
        
        print(f"\nPrompt {i+1}: {prompt[:50]}...")
        print(f"Response: {response[:100]}...")
        print(f"Tokens/sec: {tps:.2f}")
    
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    logger.info(f"Benchmark completed!")
    logger.info(f"Average tokens per second: {avg_tps:.2f}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Total tokens generated: {total_tokens}")


def main():
    parser = argparse.ArgumentParser(description="HQQ + LoRA Model Inference")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the base model or output directory")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapters (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["interactive", "benchmark", "single"],
                        help="Inference mode")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt for inference (used with --mode single)")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for nucleus sampling")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Setup HQQ
    recommended_inductor_config_setter()
    HQQLinear.set_backend(HQQBackend.PYTORCH_COMPILE)
    
    # Load model and tokenizer
    logger.info("Loading model and adapters...")
    model, tokenizer = load_model_and_adapters(
        args.model_path, 
        args.adapter_path, 
        args.device
    )
    
    logger.info("Model loaded successfully!")
    
    # Run inference based on mode
    if args.mode == "interactive":
        interactive_mode(model, tokenizer)
    
    elif args.mode == "single":
        if not args.prompt:
            logger.error("--prompt is required for single mode")
            return
        
        response, tps, gen_time = generate_text(
            model, tokenizer, args.prompt, 
            args.max_new_tokens, args.temperature, args.top_p
        )
        
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")
        print(f"Generation time: {gen_time:.2f}s")
        print(f"Tokens per second: {tps:.2f}")
    
    elif args.mode == "benchmark":
        # Default benchmark prompts
        benchmark_prompts = [
            "Explain the concept of artificial intelligence.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does machine learning work?",
            "What is the importance of data science?",
        ]
        
        benchmark_mode(model, tokenizer, benchmark_prompts, args.max_new_tokens)


if __name__ == "__main__":
    main()
