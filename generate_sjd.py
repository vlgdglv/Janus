"""
Benchmark SJD and GSD (Grouped Speculative Decoding) against baseline AR.

Usage
-----
python generate_sjd.py                 # run all methods on all prompts
python generate_sjd.py --method sjd    # SJD only
python generate_sjd.py --method gsd    # GSD only
python generate_sjd.py --method ar     # baseline AR only
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.sjd import (
    DecodeStats,
    _cfg_merge, _embed_cfg, _text_prefill,
    generate_gsd,
    generate_sjd,
    save_image,
    tokens_to_image,
)

MODEL_PATH = "/home/vlgd/Models/Janus-Pro-1B/"

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_PATH)
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

PROMPTS = [
    "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, "
    "under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
    "Three penguins in yellow construction helmets building a sandcastle on a tropical beach.",
    "A mischievous hippo playing soccer, realistic, 4k, detailed, photography",
]


def build_prompt(desc: str) -> str:
    conv = [{"role": "User", "content": desc}, {"role": "Assistant", "content": ""}]
    s = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conv, sft_format=vl_chat_processor.sft_format, system_prompt=""
    )
    return s + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate_ar(prompt: str, temperature: float = 1.0, cfg_weight: float = 5.0,
                image_token_num: int = 576, seed: int = 42):
    device    = next(vl_gpt.language_model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    stats = DecodeStats()
    past, al, ap, _ = _text_prefill(
        vl_gpt, vl_chat_processor, prompt, cfg_weight, temperature, device, stats
    )

    tokens = torch.zeros(image_token_num, dtype=torch.long, device=device)
    for i in range(image_token_num):
        t = int(torch.multinomial(ap, 1, generator=generator))
        tokens[i] = t
        emb   = _embed_cfg(vl_gpt, torch.tensor([t], dtype=torch.long, device=device))
        out   = vl_gpt.language_model.model(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past  = out.past_key_values
        stats.total_fwd += 1
        raw   = vl_gpt.gen_head(out.last_hidden_state[:, -1, :])
        al    = _cfg_merge(raw, cfg_weight)
        ap    = F.softmax(al / temperature, dim=-1)

    stats.total_tokens = image_token_num
    return tokens, stats


def run_one(method: str, prompt_str: str, out_dir: str, idx: int, seed: int = 42):
    prompt = build_prompt(prompt_str)
    os.makedirs(out_dir, exist_ok=True)

    if method == "ar":
        t0 = time.perf_counter()
        tokens, stats = generate_ar(prompt, seed=seed)
        elapsed = time.perf_counter() - t0

    elif method == "sjd":
        t0 = time.perf_counter()
        tokens, stats = generate_sjd(vl_gpt, vl_chat_processor, prompt,
                                      jacobi_window=16, seed=seed)
        elapsed = time.perf_counter() - t0

    elif method == "gsd":
        t0 = time.perf_counter()
        tokens, stats = generate_gsd(vl_gpt, vl_chat_processor, prompt,
                                      jacobi_window=16, seed=seed)
        elapsed = time.perf_counter() - t0

    else:
        raise ValueError(method)

    img = tokens_to_image(vl_gpt, tokens)
    save_image(img, f"{out_dir}/{method}_{idx:02d}.jpg")
    print(f"[{method.upper():3s}] prompt={idx}  {elapsed:.2f}s  {stats.report()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ar", "sjd", "gsd", "all"], default="all")
    parser.add_argument("--out_dir", default="generated_sjd")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    methods = ["ar", "sjd", "gsd"] if args.method == "all" else [args.method]

    for i, p in enumerate(PROMPTS):
        print(f"\nPrompt {i}: {p[:70]}...")
        for m in methods:
            run_one(m, p, args.out_dir, i, seed=args.seed)


if __name__ == "__main__":
    main()
